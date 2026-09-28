"""gunicorn settings for faffabout-web.service.

One worker with threads rather than several workers: each worker process would hold its
own DuckDB connection and boosters, and the droplet has no memory to spare. Forecasts are
dominated by the MMseqs2 subprocess, which releases the GIL, so threads serve concurrent
requests well enough.
"""
chdir = "/opt/faffabout/app"
bind = "127.0.0.1:8012"          # 8009, the local default, is taken on the droplet
workers = 1
threads = 4
timeout = 120                    # matches nginx proxy_read_timeout
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
