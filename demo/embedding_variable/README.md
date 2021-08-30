# Distributed training
```
python ev-norm-ps.py --ps_hosts=127.0.0.1:9200 --worker_hosts=127.0.0.1:9100 --task_index=0 --job_name=ps
python ev-norm-ps.py --ps_hosts=127.0.0.1:9200 --worker_hosts=127.0.0.1:9100 --task_index=0 --job_name=worker
```
