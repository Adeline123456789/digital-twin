# Correction: private calibration values in two public reports

The final task document requires raw data, private models and calibration to stay local. Commit `1f867d5a3626ba180870e275efeca4ddc1a1c8bb` mistakenly included the old and fitted device camera transforms in `experiments/spatial_cycle6/fit_001.json` and `fit_002.json`. Failed or unqualified calibration candidates are still device-specific calibration artifacts; their rejection does not justify publishing the matrices.

The current files now contain error statistics, outcomes and local artifact references/hashes. Full original artifacts remain unchanged under the local dataset. No experiment was rerun and no scientific outcome changed. The offline fitter now writes a separate `public_result.json`; console output uses that sanitized result. The publisher rejects the three observed device-transform field names in JSON/JSONL before staging. This supplements manual review; it is not a universal secret scanner. Regression tests check nested rejection and preservation of the local original, metrics and hash.

**Git history is not cleaned by this correction.** The previous public commit still contains the matrices. The task document explicitly prohibits force-push, so no published history was rewritten. Any removal from the working branch's ancestry needs an explicit exception to that rule; remote dangling commits, caches or other copies cannot be claimed erased by a branch update alone. No credential exposure was identified in these two reports, but the task's private-calibration boundary still applies.

This is correction of a publication mistake within the completed batch, not system-improvement cycle7. Six research cycles remain used, two calibration fits remain rejected, and the real-transfer objective remains unmet. See result.json for the exact local evidence hashes and bounded audit scope.

## Authorized branch-history correction

The user subsequently authorized one `force-with-lease` update on `research/right-arm-20261116` only. Replacement commit `0b75ebccb31e42bbbd175010d7a349a8206d6231` has exactly the same file tree as the reviewed corrected version, with parent `09bcfd473f074610f5b9ee59eb8e36965ad79dbf`. Both affected publication commits are absent from the updated branch ancestry. The remote SHA was verified, and local private originals remain unchanged. See `history_cleanup.json` for the observed update and scope. This does not establish removal of old GitHub commit objects, caches, forks or copies. Subsequent report publication uses an ordinary push.
