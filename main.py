from detector import *
import os

def main():
    videopath=0#'Vatsal DS entry.mp4'
    configpath=os.path.join("model_data","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelpath=os.path.join("model_data","frozen_inference_graph.pb")
    classespath=os.path.join("model_data","coco.names")
    p=detector(videopath,configpath,modelpath,classespath)
    p.onVideo()

if __name__=="__main__":
    main()