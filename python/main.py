import cv2
import numpy as np
import onnxruntime
import os
import math
from nms import multiclass_nms

dataset_classnames = {"ucf24":("Basketball", "BasketballDunk", "Biking", "CliffDiving", "CricketBowling", "Diving", "Fencing", "FloorGymnastics", "GolfSwing", "HorseRiding", "IceDancing", "LongJump", "PoleVault", "RopeClimbing", "SalsaSpin", "SkateBoarding", "Skiing", "Skijet", "SoccerJuggling", "Surfing", "TennisSwing", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog"),
                      "ava_v2.2":("bend/bow(at the waist)", "crawl", "crouch/kneel", "dance", "fall down", "get up", "jump/leap", "lie/sleep", "martial art", "run/jog", "sit", "stand", "swim", "walk", "answer phone", "brush teeth", "carry/hold (an object)", "catch (an object)", "chop", "climb (e.g. a mountain)", "clink glass", "close (e.g., a door, a box)", "cook", "cut", "dig", "dress/put on clothing", "drink", "drive (e.g., a car, a truck)", "eat", "enter", "exit", "extract", "fishing", "hit (an object)", "kick (an object)", "lift/pick up", "listen (e.g., to music)", "open (e.g., a window, a car door)", "paint", "play board game", "play musical instrument", "play with pets", "point to (an object)", "press", "pull (an object)", "push (an object)", "put down", "read", "ride (e.g., a bike, a car, a horse)", "row boat", "sail boat", "shoot", "shovel", "smoke", "stir", "take a photo", "text on/look at a cellphone", "throw", "touch (an object)", "turn (e.g., a screwdriver)", "watch (e.g., TV)", "work on a computer", "write", "fight/hit (a person)", "give/serve (an object) to (a person)", "grab (a person)", "hand clap", "hand shake", "hand wave", "hug (a person)", "kick (a person)", "kiss (a person)", "lift (a person)", "listen to (a person)", "play with kids", "push (another person)", "sing to (e.g., self, a person, a group)", "take (an object) from (a person)", "talk to (e.g., self, a person, a group)", "watch (a person)")}

class YOWOv2:
    def __init__(self, modelpath, dataset="ava_v2.2", nms_thresh=0.5, conf_thresh=0.1):
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession(modelpath)
        self.input_name = self.onnx_session.get_inputs()[0].name
        _, _, self.len_clip, self.input_height, self.input_width = self.onnx_session.get_inputs()[0].shape
        
        self.strides = [8, 16, 32]
        self.topk = 40
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        if dataset in ('ava_v2.2', 'ava'):
            self.multi_hot = True
            self.class_names = dataset_classnames['ava_v2.2']
        else:
            self.multi_hot = False
            self.class_names = dataset_classnames['ucf24']
        self.num_classes = len(self.class_names)
        
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i])) for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h,w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2) * stride
        return anchor_points

    def prepare_input(self, video_clip):
        video_clip = [cv2.resize(img, (self.input_width, self.input_height)) for img in video_clip]
        input_image = np.array(video_clip).transpose((3,0,1,2))
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
        return input_image

    def decode_boxes(self, anchors, pred_reg, stride):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        # size of bbox
        pred_box_wh = np.exp(pred_reg[..., 2:]) * stride

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)
        return pred_box
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def post_process_multi_hot(self, outputs):
        all_conf_preds, all_cls_preds, all_box_preds = [], [], []
        for i, stride in enumerate(self.strides):
            # decode box
            box_pred_i = self.decode_boxes(self.anchors[stride], outputs[6+i], stride)
            # conf pred 
            conf_pred_i = self.sigmoid(outputs[i].squeeze())   # [M,]
            # cls_pred
            cls_pred_i = self.sigmoid(outputs[3+i])                # [M, C]
            #topk
            topk_inds = np.argsort(-conf_pred_i)[:self.topk]  ###sigmoid之后全是正数,取负之后升序也就是正数时候的降序
            topk_conf_pred_i = conf_pred_i[topk_inds]
            topk_cls_pred_i = cls_pred_i[topk_inds]
            topk_box_pred_i = box_pred_i[topk_inds]
            # threshold
            keep = topk_conf_pred_i > self.conf_thresh
            topk_conf_pred_i = topk_conf_pred_i[keep]
            topk_cls_pred_i = topk_cls_pred_i[keep]
            topk_box_pred_i = topk_box_pred_i[keep]

            all_conf_preds.append(topk_conf_pred_i)
            all_cls_preds.append(topk_cls_pred_i)
            all_box_preds.append(topk_box_pred_i)

        # concatenate
        conf_preds = np.concatenate(all_conf_preds, axis=0)  # [M,]    ###objdet confindence
        cls_preds = np.concatenate(all_cls_preds, axis=0)    # [M, C]  ###class confidence
        box_preds = np.concatenate(all_box_preds, axis=0)    # [M, 4]

        # nms
        det_scores, cls_scores, bboxes = multiclass_nms(conf_preds, cls_preds, box_preds, self.nms_thresh, self.num_classes, True)
        # normalize bbox
        bboxes /= max(self.input_height, self.input_width)
        bboxes = np.clip(bboxes, 0, 1)
        return det_scores, cls_scores, bboxes
    
    def post_process_one_hot(self, outputs):
        all_det_scores, all_cls_scores, all_bboxes = [], [], []
        for i, stride in enumerate(self.strides):
            # (H x W x C,)
            det_scores_i = (np.sqrt(self.sigmoid(outputs[i]) * self.sigmoid(outputs[3+i]))).flatten()
            # Keep top k top scoring indices only.
            num_topk = min(self.topk, outputs[6+i].shape[0])

            topk_idxs = np.argsort(-det_scores_i)[:num_topk]  ###sigmoid之后全是正数,取负之后升序也就是正数时候的降序
            topk_det_scores = det_scores_i[topk_idxs]

            # filter out the proposals with low confidence score
            keep_idxs = topk_det_scores > self.conf_thresh
            det_scores = topk_det_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = np.floor_divide(topk_idxs, self.num_classes)
            cls_scores = topk_idxs % self.num_classes

            reg_pred_i = outputs[6+i][anchor_idxs]
            anchors_i = self.anchors[stride][anchor_idxs]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchors_i, reg_pred_i, stride)

            all_det_scores.append(det_scores)
            all_cls_scores.append(cls_scores)
            all_bboxes.append(bboxes)
        
        # concatenate
        det_scores = np.concatenate(all_det_scores, axis=0)  # [M,]
        cls_scores= np.concatenate(all_cls_scores, axis=0)    # [M, C]
        bboxes = np.concatenate(all_bboxes, axis=0)    # [M, 4]

        # nms
        det_scores, cls_scores, bboxes = multiclass_nms(det_scores, cls_scores, bboxes, self.nms_thresh, self.num_classes, False)
        # normalize bbox
        bboxes /= max(self.input_height, self.input_width)
        bboxes = np.clip(bboxes, 0., 1.)
        return det_scores, cls_scores, bboxes


    def detect(self, video_clip):
        orig_h, orig_w = video_clip[0].shape[:2]
        input_tensor = self.prepare_input(video_clip)

        # Perform inference on the image
        outputs = self.onnx_session.run(None, {self.input_name: input_tensor})
        outputs = [x.squeeze(axis=0) for x in outputs] ###不考虑第0维batchsize
        ###output_names = ['conf_preds0', 'conf_preds1', 'conf_preds2', 'cls_preds0', 'cls_preds1', 'cls_preds2', 'reg_preds0', 'reg_preds1', 'erg_preds2']
        
        if self.multi_hot:
            det_scores, cls_scores, bboxes = self.post_process_multi_hot(outputs)
        else:
            det_scores, cls_scores, bboxes = self.post_process_one_hot(outputs)
        
        bboxes *= np.array([[orig_w, orig_h, orig_w, orig_h]], dtype=np.float32)
        return det_scores, cls_scores, bboxes.astype(np.int32)

def vis_multi_hot(frame, det_scores, cls_scores, bboxes, class_names, vis_thresh, act_pose=False):
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i,:]
        if act_pose:
            # only show 14 poses of AVA.
            cls_conf = cls_scores[i, :14]
        else:
            # show all actions of AVA.
            cls_conf = cls_scores[i, :]
        # score = obj * cls
        det_conf = float(det_scores[i])
        det_cls_conf = np.sqrt(det_conf * cls_conf)

        indices = np.where(det_cls_conf > vis_thresh)
        scores = det_cls_conf[indices]
        indices = list(indices[0])
        scores = list(scores)

        if len(scores) > 0:
            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text
            blk   = np.zeros(frame.shape, np.uint8)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text  = []
            text_size = []

            for _, cls_ind in enumerate(indices):
                text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.5, thickness=1)[0])
                coord.append((x1+3, y1+14+20*_))
                cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-12), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
            frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)
            for t in range(len(text)):
                cv2.putText(frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1)
    
    return frame

def vis_one_hot(frame, scores, labels, bboxes, class_names, vis_thresh, text_scale=0.4):
    for i in range(bboxes.shape[0]):
        if scores[i] > vis_thresh:
            label = int(labels[i])
            text = '%s: %.2f' % (class_names[label], scores[i])

            x1, y1, x2, y2 = bboxes[i,:]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            t_size = cv2.getTextSize(text, 0, fontScale=1, thickness=2)[0]
            # plot title bbox
            cv2.rectangle(frame, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), (0,255,0), -1)
            # put the test on the title bbox
            cv2.putText(frame, text, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return frame
    

if __name__ == '__main__':
    mynet = YOWOv2('weights/yowo_v2_nano_ava.onnx', dataset='ava')
    len_clip = mynet.len_clip
    vis_thresh = 0.3

    videopath = "dataset/ucf24_demo/v_Basketball_g01_c02.mp4"
    cap = cv2.VideoCapture(videopath)
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(filename='output.mp4', fourcc=fourcc, fps=cap_fps, frameSize=(cap_width, cap_height))

    video_clip = []
    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        
        # prepare
        if len(video_clip) <= 0:
            for _ in range(len_clip):
                video_clip.append(frame)

        video_clip.append(frame)
        del video_clip[0]

        # inference
        det_scores, cls_scores, bboxes = mynet.detect(video_clip)
        if mynet.multi_hot:
            frame = vis_multi_hot(frame, det_scores, cls_scores, bboxes, mynet.class_names, vis_thresh, act_pose=False)
        else:
            frame = vis_one_hot(frame, det_scores, cls_scores, bboxes, mynet.class_names, vis_thresh)
        
        video_writer.write(frame)
        cv2.imshow('Deep learning action detection use onnxruntime', frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
