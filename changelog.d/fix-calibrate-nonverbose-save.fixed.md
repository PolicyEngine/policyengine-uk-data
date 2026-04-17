Fix `calibrate_local_areas` non-verbose branch silently failing to save weights because the `if epoch % 10 == 0` save block was indented outside the training loop.
