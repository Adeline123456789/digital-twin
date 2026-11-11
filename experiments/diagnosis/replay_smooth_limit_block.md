# Smooth real episode replay: observed model-limit incompatibility

Experiment 20261111T092440Z_a6947d13 failed before physics initialization. The underlying retained simulator raised `ValueError: State exceeds the shared robot model limits`; no replay report was produced. The wrapper preserved an infrastructure-error report rather than claiming physical failure replication.

Directly reading the first recorded real feedback shows right J2 = −0.467 degrees against the simulator lower limit 0; J3 = +0.352 degrees against simulator upper limit 0. Other right coordinates fit the inspected limits. These are recorded measurements, not newly commanded motions. Whether the difference reflects zero offset, limits, mechanics or feedback calibration remains UNKNOWN.

Do not silently clip the measured state, relax model/controller limits, or report a faithful replay after doing so. Resolve the reference/limit discrepancy with independent calibration first. Raw CSV and the initial-state hash are retained in the experiment record. This is an identified infrastructure mismatch, not the unique root cause of the original physical insertion failure.
