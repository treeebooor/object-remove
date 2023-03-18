
import copy
import cv2
import numpy as np
import torchvision.transforms as T  
from torchvision.io import read_image

class ObjectRemove():

    def __init__(self, segmentModel = None, rcnn_transforms = None, inpaintModel= None, image_path = '') -> None:
        self.segmentModel = segmentModel
        self.inpaintModel = inpaintModel
        self.rcnn_transforms = rcnn_transforms
        self.image_path = image_path
        self.highest_prob_mask = None
        self.image_orig = None
        self.image_masked = None
        self.box = None

    def run(self):
        '''
        Main run program 
        '''
        #read in image and transform
        print('Reading in image')
        images = self.preprocess_image()
        self.image_orig = images

        print("segmentation")
        #segmentation
        output = self.segment(images)
        out = output[0]

        print('user click')
        #user click
        ref_points = self.user_click()
        self.box = ref_points
        self.highest_prob_mask = self.find_mask(out, ref_points)
       
        self.highest_prob_mask[self.highest_prob_mask > 0.1]  = 1
        self.highest_prob_mask[self.highest_prob_mask <0.1] = 0
        self.image_masked = (images[0]*(1-self.highest_prob_mask))
        print('inpaint')
        #inpaint
        output = self.inpaint()
        
        #return final inpainted image
        return output

    def percent_within(self,nonzeros, rectangle):
        '''
        Calculates percent of mask inside rectangle
        '''
        rect_ul, rect_br = rectangle
        inside_count = 0
        for _,y,x in nonzeros:
            if x >= rect_ul[0] and x<= rect_br[0] and y <= rect_br[1] and y>= rect_ul[1]:
                inside_count+=1
        return inside_count / len(nonzeros)

    def iou(self, boxes_a, boxes_b):
        '''
        Calculates IOU between all pairs of boxes

        boxes_a and boxes_b are matrices with each row representing the 4 coords of a box
        '''

        x1 = np.array([boxes_a[:,0], boxes_b[:,0]]).max(axis=0)
        y1 = np.array([boxes_a[:,1], boxes_b[:,1]]).max(axis=0)
        x2 = np.array([boxes_a[:,2], boxes_b[:,2]]).min(axis=0)
        y2 = np.array([boxes_a[:,3], boxes_b[:,3]]).min(axis=0)

        w = x2-x1
        h = y2-y1
        w[w<0] = 0
        h[h<0] = 0

        intersect = w* h

        area_a = (boxes_a[:,2] - boxes_a[:,0]) * (boxes_a[:,3] - boxes_a[:,1])
        area_b = (boxes_b[:,2] - boxes_b[:,0]) * (boxes_b[:,3] - boxes_b[:,1])

        union = area_a + area_b - intersect

        return intersect / (union + 0.00001)

    def find_mask(self, rcnn_output, rectangle):
        '''
        Finds the mask with highest probability in the rectangle given
        
        '''
        bounding_boxes= rcnn_output['boxes'].detach().numpy()
        masks = rcnn_output['masks']

        ref_boxes  = np.array([rectangle], dtype=object)
        ref_boxes = np.repeat(ref_boxes, bounding_boxes.shape[0], axis=0)

        ious= self.iou(ref_boxes, bounding_boxes)

        best_ind = np.argmax(ious)

        return masks[best_ind]


        #compare masks pixelwise
        '''
        masks = rcnn_output['masks']
        #go through each nonzero point in the mask and count how many points are within the rectangles
        highest_prob_mask = None
        percent_within,min_diff = 0,float('inf')
        #print('masks lenght:', len(masks))


        for m in range(len(masks)):
            #masks[m][masks[m] > 0.5] = 255.0
            #masks[m][masks[m] < 0.5] = 0.0
            nonzeros = np.nonzero(masks[m])
            #diff = rect_area - len(nonzeros)
            p = self.percent_within(nonzeros, rectangle)
            if p > percent_within:
                highest_prob_mask = masks[m]
                percent_within = p
            print(p)
        return highest_prob_mask
        '''

    def preprocess_image(self):
        '''
        Read in image and prepare for segmentation
        '''
        img= [read_image(self.image_path)]
        _,h,w = img[0].shape
        size = min(h,w)
        if size > 512:
            img[0] = T.Resize(512, max_size=680, antialias=True)(img[0])

        images_transformed = [self.rcnn_transforms(d) for d in img]
        return images_transformed
   
    
    def segment(self,images):
        out = self.segmentModel(images)
        return out

    def user_click(self):
        '''
        Get user input for object to remove

        Returns the rectangle bounding box give by user as two points
        '''
        ref_point = []
        cache=None
        draw = False


        def click(event, x, y, flags, param):
            nonlocal ref_point,cache,img, draw
            if event == cv2.EVENT_LBUTTONDOWN:
                draw = True
                ref_point = [x, y]
                cache = copy.deepcopy(img)

            elif event == cv2.EVENT_MOUSEMOVE:
                if draw:
                    img = copy.deepcopy(cache)
                    cv2.rectangle(img, (ref_point[0], ref_point[1]), (x,y), (0, 255, 0), 2)
                    cv2.imshow('image',img)


            elif event == cv2.EVENT_LBUTTONUP:
                draw = False
                ref_point += [x,y]
                ref_point.append((x, y))
                cv2.rectangle(img, (ref_point[0], ref_point[1]), (ref_point[2], ref_point[3]), (0, 255, 0), 2)
                cv2.imshow("image", img)


        img = self.image_orig[0].permute(1,2,0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        clone = img.copy()

        cv2.namedWindow("image")

        cv2.setMouseCallback('image', click)

        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
        
            if key == ord("r"):
                img = clone.copy()
        
            elif key == ord("c"):
                break
        cv2.destroyAllWindows()
        
        return ref_point
    
    def inpaint(self):
        output = self.inpaintModel.infer(self.image_orig[0], self.highest_prob_mask, return_vals=['inpainted'])
        return output[0]



