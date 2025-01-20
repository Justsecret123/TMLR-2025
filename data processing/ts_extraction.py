import cv2 as cv
import numpy as np
import ffmpeg
import os
import glob
import subprocess
#from keras.applications.vgg16 import VGG16, preprocess_input
#from keras.models import Model
#import tensorflow as tf
import time
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch
torch.cuda.get_device_name(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
file_dir = os.path.dirname(os.path.realpath('__file__'))
scene_path = os.path.join(file_dir, 'MovieGraph', 'Shot_and_Scene', 'mg_videoinfo', 'scene_boundaries' )
videvents_path = os.path.join(file_dir, 'MovieGraph', 'Shot_and_Scene', 'mg_videoinfo', 'video_boundaries' )
movies_path = os.path.join(file_dir, 'Films' )
ref_img_path = os.path.join(file_dir, 'MovieGraph', 'Annotations', 'mg', 'startend_frame' )
clip_path = os.path.join(file_dir, "Clips")
img_path = os.path.join(file_dir, "Img_verif")
movies_list = []



def seconds_to_time(duration):
    hours = int(duration / 3600)
    minutes = int((duration % 3600) / 60)
    seconds = int(duration % 60)
    milliseconds = int(round((duration - int(duration)) * 1000))
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)

def read_lines(entry):
    with entry as f:
                output = f.readlines()
    return output

def write_output(text_path, inner_min__fr_idx, outter_min__fr_idx, ts_start, ts_end, util):
    with open(text_path, 'a') as f:
        f.write(str(inner_min__fr_idx)+" "+str(outter_min__fr_idx)+" "+str(ts_start)+" "+str(ts_end)+" "+str(util))
        f.write('\n')


def extract_scenes():
    start_time = time.time_ns()
    print("function launched")
    log_path = os.path.join(clip_path,'log.txt')
    with open(log_path, 'w') as f:
        t = (time.time_ns() - start_time) / 1E9
        f.write("%10.6f\t" %t +"lancement de la détéction")
        f.write('\n')
    for entry in os.listdir(scene_path):
        entry_split = entry.split(".")[0]
        scene_name = os.path.join(scene_path, entry_split) +".scenes.gt"
        movie_name = os.path.join(movies_path, entry_split)
        ref_img_name = os.path.join(ref_img_path, entry_split)
        videvents_name = os.path.join(videvents_path, entry_split) +".videvents"
        matidx_name = os.path.join(videvents_path, entry_split) +".matidx"
        
        glob_expression = movie_name + "*"
        file_list = glob.glob(glob_expression)
        if len(file_list) > 0:
            file_name = file_list[0]
            clips_dir = os.path.join(clip_path, entry_split)
            img_dir = os.path.join(img_path, entry_split)
            if not os.path.exists(clips_dir):
                os.mkdir(os.path.join(clips_dir))
            if not os.path.exists(img_dir):
                os.mkdir(os.path.join(img_dir))
        else:
            print("No file found for entry %s" %entry)
        if os.path.isfile(scene_name):
            scenes = open(scene_name, "r")
            videvents = open(videvents_name, "r")
            matidx = open(matidx_name, "r")

            scene_event_list = read_lines(scenes)
            videvents_list = read_lines(videvents)
            matidx_list = read_lines(matidx)
            vid_format = file_name.split(".")[-1]
            video = cv.VideoCapture(file_name, cv.CAP_FFMPEG)
            a = video.isOpened()
            fps_round = round(video.get(cv.CAP_PROP_FPS))
            fps = video.get(cv.CAP_PROP_FPS)
            print("FPS = %s" %str(fps))
            expression = os.path.join(ref_img_name, "scene-001") + "*"
            ref_img = glob.glob(expression)[0]
            ref_frame = cv.imread(ref_img)
            x, y, w, h = ref_frame.shape[1]//2, 0, ref_frame.shape[1]//2, ref_frame.shape[0]
            fr_shape = (w-5, h, 3)
            inner_frame = ref_frame[:, :x -5]
            outter_frame = ref_frame[:, x + 5 :]
            #base_model = VGG16(weights='imagenet', include_top= False, input_shape=(fr_shape[1], fr_shape[0], 3))
            model = torchvision.models.vgg16() #vgg16 = models.vgg16(pretrained=True)
            model.to(device)
            return_layers = {'30': 'out_layer30'}
            train_nodes, eval_nodes = get_graph_node_names(model)
            #print(train_nodes)
            #print("-------------------------")
            #print(eval_nodes)
            video.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, img = frame = video.read()
            if not(ret):
                print("can't load video (%s)" %str(entry_split))
                with open(log_path, 'a') as f:
                    t = (time.time_ns() - start_time) / 1E9
                    f.write("%10.6f\t" %t + "Can't open film %s" %str(entry_split))
                    f.write('\n')
                continue
            #   model = Model(inputs= base_model.input, outputs=base_model.get_layer('block5_pool').output)
            #model_with_multuple_layer = IntermediateLayerGetter(model.features, return_layers=return_layers)
            feature_extractor = create_feature_extractor(model, return_nodes=['avgpool'])

            with open(log_path, 'a') as f:
                t = (time.time_ns() - start_time) / 1E9
                f.write("-----------------------------------------")
                f.write('\n')
                f.write("%10.6f\t" %t + "Pour le film %s" %entry_split)
                f.write('\n')
            """
            cmd = [
                    "ffprobe", 
                    "-v", "quiet",
                    "-show_streams",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    "-show_entries", "stream=r_frame_rate",
                    file_name,
                ]
            result = subprocess.run(cmd,  capture_output=True)
            fps_str = result.stdout.decode().split("\n")[0]
            """     
            list_len = len(scene_event_list)
            for scene_idx in range(list_len):
                frames_idx = []
                inner_frame_dist = []
                outter_frame_dist = []
                scene_nb = scene_idx + 1
                if scene_nb  < 10:
                    expression = os.path.join(ref_img_name, "scene-00"+str(scene_nb)) + "*"
                elif scene_nb < 100:
                    expression = os.path.join(ref_img_name, "scene-0"+str(scene_nb)) + "*"
                else :
                    expression = os.path.join(ref_img_name, "scene-"+str(scene_nb)) + "*"
                print(expression)
                
                # getting start and end frame ref img  
                ref_img = glob.glob(expression)[0]
                ref_frame = cv.imread(ref_img)
                x, y, w, h = ref_frame.shape[1]//2, 0, ref_frame.shape[1]//2, ref_frame.shape[0]
                fr_shape = (w-5, h, 3)
                inner_frame = ref_frame[:, :x -5]
                outter_frame = ref_frame[:, x + 5 :]
                #cv.imwrite(output_dir+"ref_inner_%d.jpg" %scene_nb, inner_frame)
                #cv.imwrite(output_dir+"ref_outter_%d.jpg" %scene_nb, outter_frame)

                #x_ref_inner = tf.keras.utils.img_to_array(inner_frame)
                #x_ref_inner = np.expand_dims(x_ref_inner, axis = 0)
                #x_ref_inner = preprocess_input(x_ref_inner)
                transform = torchvision.transforms.ToTensor() 
                features_inner = feature_extractor(transform(inner_frame).cuda())['avgpool']#model.features[31](torch.from_numpy(inner_frame))
                #features_inner = model_with_multuple_layer(inner_frame) #model.predict(x_ref_inner)
                #x_ref_outter = tf.keras.utils.img_to_array(outter_frame)
                #x_ref_outter = np.expand_dims(x_ref_outter, axis = 0)
                #x_ref_outter = preprocess_input(x_ref_outter)
                features_outter = feature_extractor(transform(outter_frame).cuda())['avgpool'] #model.features[31](torch.from_numpy(outter_frame)) #model_with_multuple_layer(outter_frame) #model.predict(x_ref_outter)
                features_inner = features_inner.flatten().cpu().detach().numpy()
                features_outter = features_outter.flatten().cpu().detach().numpy()

                videvent_idx_start = int(scene_event_list[scene_idx].split(" ")[0]) 
                videvent_idx_end = int(scene_event_list[scene_idx].split(" ")[1])
                util = int(scene_event_list[scene_idx].split(" ")[2])
                if videvent_idx_start > 1 :
                    #videvent_idx_start -= 1
                    videvent_idx_start -= 2
                elif videvent_idx_start == 1:
                    videvent_idx_start -= 1
                if videvent_idx_end < len(videvents_list) - 1:
                    videvent_idx_end += 1
                    fr_end = int(videvents_list[videvent_idx_end].split(" ")[0])
                else :
                    fr_end = int(matidx_list[-1].split(" ")[0])
                if videvent_idx_start == 0:
                    fr_start = 0
                else :
                    fr_start = int(videvents_list[videvent_idx_start].split(" ")[0])
                #if videvents_list[videvent_idx_end] == "":
                #    continue
                #else: 
                
                for fr_idx in range(fr_start, fr_end, 8):
                    #if fr_idx != 0:
                    video.set(cv.CAP_PROP_POS_FRAMES, fr_idx)
                    ret, img = frame = video.read()
                    if not(ret):
                        print("cannot open for idx %d" %fr_idx)
                        with open(log_path, 'a') as f:
                            t = (time.time_ns() - start_time) / 1E9
                            f.write("%10.6f\t" %t + "Can't open idx %d" %str(fr_idx))
                            f.write('\n')
                        continue
                    img = cv.resize(img, (fr_shape[0], fr_shape[1]), interpolation = cv.INTER_NEAREST)
                    
        
                    #x_img = tf.keras.utils.img_to_array(img)
                    #x_img = np.expand_dims(x_img, axis = 0)
                    #x_img = preprocess_input(x_img)
                    #features_img = model.predict(x_img)
                    #features_img = features_img.flatten()
                    features_img = feature_extractor(transform(img).cuda())['avgpool']#model.features[31](torch.from_numpy(img)) #model_with_multuple_layer(img)
                    features_img = features_img.flatten().cpu().detach().numpy()
                    #cosine_inner = np.dot(features_inner,features_img)/(np.linalg.norm(features_inner)*np.linalg.norm(features_img))
                    #cosine_outter = np.dot(features_outter,features_img)/(np.linalg.norm(features_outter)*np.linalg.norm(features_img))
                    dist_inner = np.linalg.norm(features_inner - features_img)
                    dist_outter = np.linalg.norm(features_outter - features_img)
                    frames_idx.append(fr_idx)
                    inner_frame_dist.append(dist_inner)
                    outter_frame_dist.append(dist_outter)
                    if (fr_idx - fr_start) % 300 == 0:
                        print("frame #%d dist for inner = %f and for outter is %f " %(fr_idx, dist_inner, dist_outter))
                        #cv.imwrite(clips_dir+"frame_%d.jpg" %fr_idx, img)
                
                inner_min_dist_idx = np.argsort(inner_frame_dist)[0]
                inner_min__fr_idx = frames_idx[inner_min_dist_idx]
                
                outter_frame_dist_new = outter_frame_dist[inner_min_dist_idx:]
                frames_idx_new = frames_idx[inner_min_dist_idx:]
                #print("inner idx is %d so outter can't be inferior  ( old size is %d new size is %d )" %(inner_min__fr_idx, len(outter_frame_dist) , len(outter_frame_dist_new)))
                outter_min_dist_idx = np.argsort(outter_frame_dist_new)[0]
                outter_min__fr_idx = frames_idx_new[outter_min_dist_idx]
               
                print("#%d best fit for start frame is #%d and for last frame is #%d " %(scene_idx, inner_min__fr_idx, outter_min__fr_idx))
                #print("Minimal distance for inner is %f and for outter is %f " %(min(inner_frame_dist), min(outter_frame_dist)))
                #print("for scene idx %d it's over" %scene_idx)
                
                ts_start = seconds_to_time(inner_min__fr_idx/fps)
                ts_end = seconds_to_time(outter_min__fr_idx/fps)
                if util :
                    vid_name = "%s_scene_%d.%s"%(entry_split,scene_nb,vid_format)
                    vid_path = os.path.join(clips_dir,vid_name)
                    cmd = [
                    "ffmpeg",
                    "-i", file_name,
                    "-c:v", "h264_nvenc", 
                    "-ss", ts_start,
                    "-to", ts_end,
                    
                    vid_path ]
                    print(" ".join(cmd))
                    subprocess.run(cmd)
                    """
                    inner_name = "res_inner_%d_%s.jpg" %(scene_nb, img_dir)
                    res_path = os.path.join(img_dir, inner_name)
                    video.set(cv.CAP_PROP_POS_FRAMES, inner_min__fr_idx)
                    ret, img  = video.read()
                    cv.imwrite(res_path , img)
                    outter_name = "res_inner_%d_%s.jpg" %(scene_nb, img_dir)
                    res_path = os.path.join(img_dir, outter_name)
                    video.set(cv.CAP_PROP_POS_FRAMES, outter_min__fr_idx)
                    ret, img  = video.read()
                    cv.imwrite(res_path, img)
                    """

                with open(log_path, 'a') as f:
                    t = (time.time_ns() - start_time) / 1E9
                    f.write("%10.6f\t" %t + "Pour la scene #%s (idx %d)" %(scene_nb, scene_idx))
                    f.write('\n')
                    f.write("%10.6f\t" %t + "best fit for start frame is #%d and for last frame is #%d " %(inner_min__fr_idx, outter_min__fr_idx))
                    f.write("%10.6f\t" %t + str(inner_min__fr_idx)+" "+str(outter_min__fr_idx)+" "+str(ts_start)+" "+str(ts_end)+" "+str(util))
                    f.write('\n')
                text_path = os.path.join(clips_dir,'scenes.txt')
                with open(text_path, 'a') as f:
                    f.write(str(inner_min__fr_idx)+" "+str(outter_min__fr_idx)+" "+str(ts_start)+" "+str(ts_end)+" "+str(util))
                    f.write('\n')
                #write_output(text_path, inner_min__fr_idx, outter_min__fr_idx, ts_start, ts_end, util)
                    
            # reshaping frame
            #inner_ref = np.reshape(inner_frame, fr_shape)
            #outter_ref = np.reshape(outter_frame, fr_shape)

            # opening 1 frame every 2sec 
            #model = VGG16(weights='imagenet', include_top= False)
            # input_shape=(160,320,3)
            
            #inp2 = base_model.input
            #out2 = base_model.output
            
            
            
            #print(block4_pool_features.shape)
            #print(block4_pool_features)


if __name__=="__main__":
    print("launching programm")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    #print(cv2.cuda.getCudaEnabledDeviceCount())
    extract_scenes()