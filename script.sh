#!/bin/bash

# llc -mtriple=dlc -filetype=obj --scheditins=false --optimize-regalloc=true -O0 $1 \
#  |& grep "Unrecognized relocation type"

# curl 'https://oapi.dingtalk.com/robot/send?access_token=5e773ecb4c8c643a3fe16df50dc260b3220bf425033503f5bc4cca01421f1ec9' \
#  -H 'Content-Type: application/json' \
#  -d '{"msgtype": "markdown", "markdown": {"title":"点进来看看我","text":"####好好好 \n > 点下面直达百度 \n > ![screenshot](https://img2020.cnblogs.com/blog/1957096/202005/1957096-20200527110106198-1974765350.jpg)\n > ##### 百度 [baidu](https://www.baidu.com/) \n"}, "at":{"atUserIds":["白牛"],"isAtAll":false}}'

# https://oapi.dingtalk.com/robot/send?access_token=5e875e8726b5c3f741ed783288c0dcfc2b5c9fe7c976c57645c3c7a7465977ab
curl 'https://oapi.dingtalk.com/robot/send?access_token=5e773ecb4c8c643a3fe16df50dc260b3220bf425033503f5bc4cca01421f1ec9'  \
 -H 'Content-Type: application/json' \
 -d '{"msgtype": "text","text": {"content":"我就是我, 是不一样的烟火 \n 试一下能不能@all"},"at":{"atMobiles":[""],"isAtAll":false}}'