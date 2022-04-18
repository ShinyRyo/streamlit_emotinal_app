import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import requests
import os.path
import numpy as np
from openvino.inference_engine import IECore

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像認識アプリ")
st.sidebar.write("オリジナルの画像認識モデルを使って何の画像かを判定します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

# IEコアの初期化
ie = IECore()

# #モデルの準備（顔検出） 
# model_face = 'intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml'
# weights_face = 'intel/face-detection-retail-0004/FP32/face-detection-retail-0004.bin'

# #モデルの準備（感情分類） 
# model_emotion = 'intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml'
# weights_emotion = 'intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin'

# # モデルの読み込み（顔検出） 
# net_face = ie.read_network(model=model_face, weights=weights_face)
# exec_net_face = ie.load_network(network=net_face, device_name='CPU')

# # モデルの読み込み（感情分類） 
# net_emotion = ie.read_network(model=model_emotion, weights=weights_emotion)
# exec_net_emotion = ie.load_network(network=net_emotion, device_name='CPU')

# # 入出力データのキー取得 
# input_blob_face = next(iter(net_face.input_info))
# out_blob_face = next(iter(net_face.outputs))

# input_blob_emotion = next(iter(net_emotion.input_info))
# out_blob_emotion = next(iter(net_emotion.outputs))

# frame = Image.open(img_file)
# frame = np.array(frame, dtype=np.uint8)
# img = read_img(img_file)

# # 推論実行 
# out = exec_net_face.infer(inputs={input_blob_face: img})

# # 出力から必要なデータのみ取り出し 
# out = out[out_blob_face]
# out = np.squeeze(out) #サイズ1の次元を全て削除 

# # 検出されたすべての顔領域に対して１つずつ処理 
# for detection in out:
#     # conf値の取得 
#     confidence = float(detection[2])

#     # バウンディングボックス座標を入力画像のスケールに変換 
#     xmin = int(detection[3] * frame.shape[1])
#     ymin = int(detection[4] * frame.shape[0])
#     xmax = int(detection[5] * frame.shape[1])
#     ymax = int(detection[6] * frame.shape[0])

#     # conf値が0.5より大きい場合のみバウンディングボックス表示 
#     if confidence > 0.5:
#            # 顔検出領域はカメラ範囲内に補正する。特にminは補正しないとエラーになる 
#             if xmin < 0:
#                 xmin = 0
#             if ymin < 0:
#                 ymin = 0
#             if xmax > frame.shape[1]:
#                 xmax = frame.shape[1]
#             if ymax > frame.shape[0]:
#                 ymax = frame.shape[0]
 
#             # 顔領域のみ切り出し 
#             frame_face = frame[ ymin:ymax, xmin:xmax ]
 
#             # 入力データフォーマットへ変換 
#             img = np.resize(frame,(64,64,3))
#             img = img.transpose((2, 0, 1))    # HWC > CHW 
#             img = np.expand_dims(img, axis=0) # 次元合せ 
 
#             # 推論実行 
#             out = exec_net_emotion.infer({input_blob_emotion: img})
 
#             # 出力から必要なデータのみ取り出し 
#             out = out[out_blob_emotion]
#             out = np.squeeze(out) #不要な次元の削減 
 
#             # 出力値が最大のインデックスを得る 
#             index_max = np.argmax(out)
 
#             # 各感情の文字列をリスト化 
#             list_emotion = ['neutral', 'happy', 'sad', 'surprise', 'anger']
 
#             # バウンディングボックス表示 
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)

#             # 文字列描画 
#             cv2.putText(frame, list_emotion[index_max], (300, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) 
 
#             # １つの顔で終了 
#             break

# # 画像表示
# fig, ax = plt.subplots()
# ax.imshow(frame)