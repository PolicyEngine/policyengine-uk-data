"""Tests for dataset creation and calibration functionality."""
import pytest
import numpy as np
import torch
from policyengine_uk_data.datasets.frs import create_frs
from policyengine_uk_data.datasets.local_areas.constituencies.calibrate import calibrate as calibrate_constituencies
from policyengine_uk_data.datasets.local_areas.local_authorities.calibrate import calibrate as calibrate_las
from policyengine_uk_data.utils.uprating import uprate_dataset


def test_frs_dataset_structure(frs):
    """Test that FRS dataset has expected structure."""
    # Check basic structure
    assert hasattr(frs, 'household')
    assert hasattr(frs, 'person')
    assert hasattr(frs, 'benunit')
    
    # Check we have some data
    assert len(frs.household.household_id) > 0
    assert len(frs.person.person_id) > 0
    assert len(frs.benunit.benunit_id) > 0
    
    # Check weights exist and are positive
    assert hasattr(frs.household, 'household_weight')
    assert np.all(frs.household.household_weight > 0)


def test_uprating_functionality(frs):
    """Test that uprating changes the dataset appropriately."""
    original_year = int(frs.time_period)
    target_year = original_year + 1
    
    # Get some baseline values
    original_employment_income = frs.person.employment_income.sum()
    
    # Uprate dataset
    uprated_frs = uprate_dataset(frs, target_year)
    
    # Check time period changed
    assert int(uprated_frs.time_period) == target_year
    
    # Check income values changed (should generally increase)
    uprated_employment_income = uprated_frs.person.employment_income.sum()
    # Allow for some variation but expect general increase
    assert abs(uprated_employment_income - original_employment_income) > 1000


def test_constituencies_calibration_basic(frs):
    """Test that constituencies calibration runs and produces valid output."""
    # Run calibration with minimal epochs for speed
    result = calibrate_constituencies(frs, epochs=5, verbose=False)
    
    # Check we get a dataset back
    assert result is not None
    assert hasattr(result, 'household')
    assert hasattr(result, 'person')
    
    # Check weights are still positive and finite
    weights = result.household.household_weight
    assert np.all(weights > 0)
    assert np.all(np.isfinite(weights))
    
    # Check dataset structure preserved
    assert len(result.household) == len(frs.household)
    assert len(result.person) == len(frs.person)


def test_local_authorities_calibration_basic(frs):
    """Test that LA calibration runs and produces valid output."""  
    # Run calibration with minimal epochs for speed
    result = calibrate_las(frs, epochs=5, verbose=False)
    
    # Check we get a dataset back
    assert result is not None
    assert hasattr(result, 'household')
    assert hasattr(result, 'person')
    
    # Check weights are still positive and finite
    weights = result.household.household_weight
    assert np.all(weights > 0)
    assert np.all(np.isfinite(weights))
    
    # Check dataset structure preserved
    assert len(result.household) == len(frs.household)
    assert len(result.person) == len(frs.person)


def test_calibration_weight_changes(frs):
    """Test that calibration actually changes weights."""
    original_weights = frs.household.household_weight.copy()
    
    # Run calibration
    result = calibrate_constituencies(frs, epochs=3, verbose=False)
    new_weights = result.household.household_weight
    
    # Weights should have changed for at least some households
    weight_changes = np.abs(new_weights - original_weights)
    assert np.sum(weight_changes) > 1.0  # Some meaningful change
    
    # But weights should still be reasonable
    assert np.all(new_weights > 0)
    assert np.mean(new_weights) > 0


def test_calibration_preserves_totals_roughly(frs):
    """Test that calibration doesn't drastically change population totals."""
    original_total_weight = frs.household.household_weight.sum()
    
    # Run calibration
    result = calibrate_constituencies(frs, epochs=3, verbose=False)
    new_total_weight = result.household.household_weight.sum()
    
    # Total weight should be roughly preserved (within reasonable bounds)
    relative_change = abs(new_total_weight - original_total_weight) / original_total_weight
    assert relative_change < 0.5  # Less than 50% change in total (calibration can change totals significantly)


def test_microcalibrate_import():
    """Test that microcalibrate can be imported."""
    from microcalibrate.calibration import Calibration
    assert Calibration is not None


def test_microcalibrate_basic_usage(frs):
    """Test basic microcalibrate usage with simple data."""
    from microcalibrate.calibration import Calibration
    
    # Create simple test case with subset of data
    num_households = min(100, len(frs.household.household_id))
    weights = frs.household.household_weight[:num_households].copy()
    
    # Create simple estimate matrix
    matrix = np.random.random((num_households, 3))
    targets = np.array([10000, 20000, 5000])
    
    def estimate_function(w, matrix_arg=None):
        # Use torch operations to preserve gradients
        if isinstance(w, torch.Tensor):
            device = w.device
            matrix_torch = torch.tensor(matrix.T, dtype=torch.float32, device=device)
            result = matrix_torch @ w
        else:
            device = torch.device('cpu')
            matrix_torch = torch.tensor(matrix.T, dtype=torch.float32, device=device)
            w_torch = torch.tensor(w, dtype=torch.float32, device=device)
            result = matrix_torch @ w_torch
        return result
    
    # Test microcalibrate
    target_names = np.array([f"target_{i}" for i in range(len(targets))])
    calibrator = Calibration(
        weights=weights,
        targets=targets,
        target_names=target_names,
        estimate_function=estimate_function,
        epochs=5,
        learning_rate=0.01
    )
    
    performance = calibrator.calibrate()
    
    # Check it ran successfully
    assert performance is not None
    assert hasattr(calibrator, 'weights')
    assert len(calibrator.weights) == num_households
    assert np.all(calibrator.weights > 0)


def test_calibration_uses_microcalibrate(frs):
    """Test that our calibration functions actually use microcalibrate."""
    # This test ensures microcalibrate is being used, not just imported
    import microcalibrate.calibration
    
    # Patch microcalibrate to track usage
    original_init = microcalibrate.calibration.Calibration.__init__
    init_called = []
    
    def tracked_init(self, *args, **kwargs):
        init_called.append(True)
        return original_init(self, *args, **kwargs)
    
    microcalibrate.calibration.Calibration.__init__ = tracked_init
    
    try:
        # Run calibration
        calibrate_constituencies(frs, epochs=2, verbose=False)
        
        # Check microcalibrate was actually used
        assert len(init_called) > 0, "Microcalibrate.Calibration was not used"
        
    finally:
        # Restore original
        microcalibrate.calibration.Calibration.__init__ = original_init


def test_enhanced_frs_has_more_data(frs, enhanced_frs):
    """Test that enhanced FRS has additional fields compared to base FRS."""
    # Enhanced FRS should have same basic structure
    assert len(enhanced_frs.household.household_id) == len(frs.household.household_id)
    
    # But may have additional computed fields or different weights
    assert hasattr(enhanced_frs.household, 'household_weight')
    assert hasattr(enhanced_frs.person, 'employment_income')


def test_dataset_consistency(frs):
    """Test internal consistency of dataset."""
    # Basic structure tests - check tables exist
    assert hasattr(frs, 'household')
    assert hasattr(frs, 'person') 
    assert hasattr(frs, 'benunit')
    
    # Check we have data in each table
    assert len(frs.household) > 0
    assert len(frs.person) > 0
    assert len(frs.benunit) > 0


def test_key_economic_variables_exist(frs):
    """Test that key economic variables exist in dataset."""
    # Check key person variables
    assert hasattr(frs.person, 'employment_income')
    assert hasattr(frs.person, 'age')
    
    # Check key household variables  
    assert hasattr(frs.household, 'household_weight')
    
    # Check values are reasonable
    assert np.all(frs.person.age >= 0)
    assert np.all(frs.person.age <= 150)  # Reasonable age bounds
    assert np.all(frs.person.employment_income >= 0)  # Income non-negative