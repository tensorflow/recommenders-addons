ps -ef|grep "bpv2.py"|grep -v grep|awk '{print $2}'| xargs kill -9
sleep 1
