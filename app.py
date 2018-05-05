#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2


def main():
    # 入力画像の読み込み
    img = cv2.imread("test.png")

    # カスケード型識別器の読み込み
    # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    # cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
    # cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
    # cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
    # cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
    # cascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
    # cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔領域の探索
    face = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30))

    # 顔領域を赤色の矩形で囲む
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y+h), (0, 0, 200), 3)

    # 結果を出力
    cv2.imshow("result.", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
