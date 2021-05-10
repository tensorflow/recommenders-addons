ps -ef|grep "movielens-"|grep -v grep|awk '{print $2}'| xargs kill -9
sleep 1
echo "result"
ps -ef|grep "movielens-"
