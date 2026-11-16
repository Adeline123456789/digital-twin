# Attempt 5 startup failure

No task motion started. Old process exited between the procfs existence check and read, producing ProcessLookupError. Local minimal fix treats ESRCH/ENOENT as process exit and retains the subsequent lock and hardware checks. Raw capture is preserved outside formal Episode roots.
