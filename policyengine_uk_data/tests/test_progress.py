from policyengine_uk_data.utils.progress import ProcessingProgress


def test_track_dataset_creation_logs_in_ci(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    progress = ProcessingProgress()

    with progress.track_dataset_creation(["Build base", "Save final"]) as (
        update_dataset,
        nested_progress,
    ):
        assert nested_progress is None
        update_dataset("Build base", "processing")
        update_dataset("Build base", "completed")
        update_dataset("Save final", "processing")
        update_dataset("Save final", "completed")

    output = capsys.readouterr().out
    assert "[dataset] starting: Build base" in output
    assert "[dataset] completed (1/2): Build base" in output
    assert "[dataset] completed (2/2): Save final" in output


def test_track_calibration_logs_heartbeats_in_ci(monkeypatch, capsys):
    monkeypatch.setenv("CI", "true")

    progress = ProcessingProgress()

    with progress.track_calibration(12) as update_calibration:
        for iteration in range(1, 13):
            update_calibration(iteration, calculating_loss=True)
            update_calibration(iteration, loss_value=iteration / 10)

    output = capsys.readouterr().out
    assert "[calibration] epoch 1/12: calculating loss" in output
    assert "[calibration] epoch 10/12: loss=1.000000" in output
    assert "[calibration] epoch 12/12: loss=1.200000" in output
