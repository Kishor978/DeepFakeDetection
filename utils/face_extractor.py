import os
import cv2
import numpy as np
import torch

class FaceExtractor:
    """Face Extraction Work flow"""
    def __init__(self,video_read_function,facedetctor):
        """Creates a new FaceExtractor.

        Arguments:
            video_read_fn: a function that takes in a path to a video file
                and returns a tuple consisting of a NumPy array with shape
                (num_frames, H, W, 3) and a list of frame indices, or None
                in case of an error
            facedet: the face detector object
        """
        self.video_read_function=video_read_function
        self.facedetctor=facedetctor
    
    def process_videos(self, input_dir, filesnames,video_index):
        """Grabs frames from videos and tries to find faces in each frames
        The frames are split into tiles, and the tiles from the different videos 
        are concatenated into a single batch. This means the face detector gets
        a batch of size len(video_idxs) * num_frames * num_tiles (usually 3).

        Arguments:
            input_dir: base folder where the video files are stored
            filenames: list of all video files in the input_dir
            video_idxs: one or more indices from the filenames list; these
                are the videos we'll actually process
        """
        target_size=self.facedetctor.input_size
        
        videos_read = []
        frames_read = []
        frames = []
        tiles = []
        resize_info = []
        
        for video_idx in video_index:
            # read full frame from video
            filesname=filesnames[video_index]
            video_path=os.path.join(input_dir,filesname)
            result=self.video_read_function(video_path)
            # skip video if error is found
            if result is None: continue
            
            videos_read.append(video_idx)
            my_frames,my_index= result
            frames.append(my_frames)
            frames_read.append(my_index)
            # Split the frames into several tiles. Resize the tiles to 128x128.
            my_tiles, my_resize_info = self._tile_frames(my_frames, target_size)
            tiles.append(my_tiles)
            resize_info.append(my_resize_info)
            
        # put all tiles for all frames into a single batch
        batch=np.concatenate(tiles)
        # Run the face detector. The result is a list of PyTorch tensors, one for each image in the batch.
        all_detections = self.facedetctor.predict_on_batch(batch, apply_nms=False)

        result = []
        offs = 0
        for v in range(len(tiles)):
            # Not all videos may have the same number of tiles, so find which 
            # detections go with which video.
            num_tiles = tiles[v].shape[0]
            detections = all_detections[offs:offs + num_tiles]
            offs += num_tiles

            # Convert the detections from 128x128 back to the original frame size.
            detections = self._resize_detections(detections, target_size, resize_info[v])

            # Because we have several tiles for each frame, combine the predictions
            # from these tiles. The result is a list of PyTorch tensors, but now one
            # for each frame (rather than each tile).
            num_frames = frames[v].shape[0]
            frame_size = (frames[v].shape[2], frames[v].shape[1])
            detections = self._untile_detections(num_frames, frame_size, detections)

            # The same face may have been detected in multiple tiles, so filter out
            # overlapping detections. This is done separately for each frame.
            detections = self.facedet.nms(detections)
            for i in range(len(detections)):
                # Crop the faces out of the original frame.
                faces = self._add_margin_to_detections(detections[i], frame_size, 0.2)
                faces = self._crop_faces(frames[v][i], faces)

                # Add additional information about the frame and detections.
                scores = list(detections[i][:, 16].cpu().numpy())
                frame_dict = { "video_idx": videos_read[v],
                               "frame_idx": frames_read[v][i],
                               "frame_w": frame_size[0],
                               "frame_h": frame_size[1],
                               "faces": faces, 
                               "scores": scores }
                result.append(frame_dict)
                
        return result
            