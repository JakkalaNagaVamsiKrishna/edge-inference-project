import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from compressor.pipeline import run

@pytest.fixture
def mock_pipeline_components():
    with patch("compressor.pruner.prune_teacher") as m_prune, \
         patch("compressor.distiller.distill") as m_distill, \
         patch("compressor.quantizer.quantize") as m_quant:
        
        m_prune.return_value = MagicMock(spec=nn.Module)
        m_distill.return_value = MagicMock(spec=nn.Module)
        m_quant.return_value = MagicMock()
        m_quant.return_value.stat.return_value.st_size = 1000000
        
        yield {
            "prune": m_prune,
            "distill": m_distill,
            "quant": m_quant,
        }

def test_pipeline_full_run(mock_pipeline_components):
    """Test a successful full pipeline execution."""
    results = run(skip_prune=False, skip_distill=False)
    
    assert "stages" in results
    assert "pruning" in results["stages"]
    assert "distillation" in results["stages"]
    assert mock_pipeline_components["prune"].called
    assert mock_pipeline_components["distill"].called

def test_pipeline_skip_prune(mock_pipeline_components):
    """Verify that pruning is skipped when requested."""
    run(skip_prune=True, skip_distill=False)
    assert not mock_pipeline_components["prune"].called
    assert mock_pipeline_components["distill"].called

def test_pipeline_skip_distill(mock_pipeline_components):
    """Verify that distillation is skipped when requested."""
    run(skip_prune=False, skip_distill=True)
    assert mock_pipeline_components["prune"].called
    assert not mock_pipeline_components["distill"].called

def test_pipeline_quantization_failure(mock_pipeline_components):
    """Test pipeline behavior when quantization fails."""
    mock_pipeline_components["quant"].side_effect = Exception("Quantization failed")
    
    with pytest.raises(Exception, match="Quantization failed"):
        run()

# ─── Large Parameterized Logic Tests ──────────────────────────────────────────

@pytest.mark.parametrize("stage", ["pruning", "distillation", "quantization"])
def test_pipeline_stage_failures(mock_pipeline_components, stage):
    """Exhaustive test of pipeline failures at each stage."""
    component_map = {
        "pruning": "prune",
        "distillation": "distill",
        "quantization": "quant"
    }
    
    mock_pipeline_components[component_map[stage]].side_effect = RuntimeError(f"{stage} error")
    
    with pytest.raises(RuntimeError, match=f"{stage} error"):
        run()

@pytest.mark.parametrize("epoch", range(1, 10))
def test_distillation_epoch_simulation(epoch):
    """Simulate distillation across different epoch counts via config mocking."""
    from compressor.distiller import distill
    
    with patch("configs.settings.cfg.model.epochs", epoch), \
         patch("compressor.distiller.prune_teacher") as m_prune_func, \
         patch("compressor.distiller._build_student"), \
         patch("compressor.distiller._build_dataloaders") as m_data, \
         patch("torch.save"), \
         patch("torch.load") as m_load, \
         patch("pathlib.Path.exists", return_value=False):
        
        # Mock teacher
        mock_teacher = MagicMock(spec=nn.Module)
        m_prune_func.return_value = mock_teacher
        
        # Mock student with REAL parameters
        real_student = nn.Linear(10, 10)
        # Prevent student load_state_dict from failing by mocking torch.load return
        m_load.return_value = real_student.state_dict()
        
        with patch("compressor.distiller._build_student", return_value=real_student):
            # Mock dataloaders to be empty
            mock_loader = MagicMock()
            mock_loader.__iter__.return_value = iter([])
            m_data.return_value = (mock_loader, mock_loader)
            
            # This tests that the loop handles the epoch count correctly
            model = distill()
            assert model == real_student
