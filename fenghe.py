#! -*- coding: utf-8 -*-
"""
Author: ZhenYuSha
Create Time: 2019-1-14
Info: Websocket 的使用示例
"""
import asyncio
import websockets
from gluoncv import data, utils
from mxnet import gluon

import mxnet as mx
import cv2
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
classes = ['hat', 'person']
ctx = mx.cpu()
websocket_users = set()

# 接收客户端消息并处理，这里只是简单把客户端发来的返回回去
async def recv_user_msg(websocket):
    while True:
        #要求传来一帧画面
        recv_text = await websocket.recv()
        print("recv_text:", websocket.pong, recv_text)
        x, frame = data.transforms.presets.yolo.load_test(recv_text, short=416)
        x = x.as_in_context(ctx)
        net = gluon.SymbolBlock.imports(symbol_file='./darknet53-symbol.json', input_names=['data'],
                                            param_file='./darknet53-0000.params', ctx=ctx)

        box_ids, scores, bboxes = net(x)

        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()

        if isinstance(box_ids, mx.nd.NDArray):
            box_ids = box_ids.asnumpy()

        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()

        length = scores[0]
        # 获得1帧内预测个数
        num = 0;
        for i in range(0, len(length)):
            if scores[0][i] != -1:
                num += 1
            else:
                break
        id = 'id'
        for j in range(0, num):
            id += ' ' + str(box_ids[0][j][0])
            id += ' ' + str(scores[0][j][0])
        ax = utils.viz.cv_plot_bbox(frame, bboxes[0], scores[0], box_ids[0], class_names=classes, thresh=0.4)
        cv2.imwrite(recv_text, frame)
        await websocket.send(recv_text + ' ' + id)



# 服务器端主逻辑
async def run(websocket, path):
    while True:
        try:
            #await check_user_permit(websocket)
            await recv_user_msg(websocket)
        except websockets.ConnectionClosed:
            print("ConnectionClosed...", path)  # 链接断开
            print("websocket_users old:", websocket_users)
            websocket_users.remove(websocket)
            print("websocket_users new:", websocket_users)
            break
        except websockets.InvalidState:
            print("InvalidState...")  # 无效状态
            break
        except Exception as e:
            print("Exception:", e)


if __name__ == '__main__':
    print("127.0.0.1:8181 websocket...")
    asyncio.get_event_loop().run_until_complete(websockets.serve(run, "127.0.0.1", 8181))
    asyncio.get_event_loop().run_forever()