import cv2
import torch
import math

# parameter
PATH_VIDEO = './trafficvideo/GOPR6163.MP4'
window_name = 'frame'
delay = 1
p_distance = 10     # px
p_distance_x = 100
p_distance_y = 25
crossing_boarder_x = 640
thresh_confidence = 0.15


class Object:
    def __init__(self, id_arg, x=0, y=0, width=0, height=0):
        self.id = id_arg
        self.x = x
        self.y = y
        self.x_k1 = x
        self.y_k1 = y
        self.dx = 0
        self.dy = 0
        self.width = width
        self.height = height
        self.distance_min = 10000
        self.counter_no_detection = 0
        self.line_crossed = False
        self.x_close = 10000
        self.y_close = 10000
        self.width_close = width
        self.height_close = height
        self.distance_close = 10000

    def set_distance_min(self, dist_x, dist_y):
        distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
        self.distance_min = min(self.distance_min, distance)

    def reset_distance_min(self):
        self.distance_min = 10000

    def count_no_detection(self):
        self.counter_no_detection = self.counter_no_detection + 1

    def reset_no_detection_counter(self):
        self.counter_no_detection = 0

    def delete_object_counter(self, id_del_list):
        if self.counter_no_detection > 10:
            self.delete_object(id_del_list)

    def delete_object(self, id_del_list):
        if not self.id in id_del_list:
            id_del_list.append(self.id)

    def distance(self, x, y):
        distance_obj = math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
        return distance_obj

    def update_object(self, x, y, width, height):
        self.x_k1 = self.x
        self.y_k1 = self.y
        self.x = x
        self.y = y
        self.dx = self.x - self.x_k1
        self.dy = self.y - self.y_k1
        self.width = width
        self.height = height
        self.reset_no_detection_counter()

    def check_crossing_border_left(self, x_bound):
        crossing_left = False

        if (self.x_k1 > x_bound) and (self.x < x_bound):
            crossing_left = True

        if self.line_crossed:
            crossing_left = False

        if crossing_left:
            self.line_crossed = True

        return crossing_left

    def check_crossing_border_right(self, x_bound):
        crossing_right = False

        if (self.x_k1 < x_bound) and (self.x > x_bound):
            crossing_right = True

        if self.line_crossed:
            crossing_right = False

        if crossing_right:
            self.line_crossed = True

        return crossing_right

    def check_close_object(self, x, y, width, height):
        # check if the new detection is closer than exiting.
        if self.distance(x, y) < self.distance_close:
            # update close object info
            self.x_close = x
            self.y_close = y
            self.width_close = width
            self.height_close = height


# check the position is accepted for object creation
def in_position_object_creation(x, y, object_dict):
    x_min = 0
    x_max = 1280
    y_min = 0
    y_max = 720

    if x_min < x < x_max and y_min < y < y_max:
        in_position = True
    else:
        in_position = False

    return in_position


# check the position is accepted for object deletion
def in_position_object_deletion(x, y):
    x_min = 0
    x_max = 1280
    y_min = 0
    y_max = 720

    if x_min < x < x_max and y_min < y < y_max:
        in_position = True
    else:
        in_position = False

    return in_position


if __name__ == '__main__':
    # initialization
    i_object = {}       # list of control object
    id_delete = []      # id list of object to be deleted
    id_t = 0            # control object id counter
    counter_traffic = 0 # traffic counter

    # load model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    # load video file
    video_file = cv2.VideoCapture(PATH_VIDEO)

    while True:
        # read video
        ret, frame = video_file.read()

        if ret:
            id_detect = 0
            detected_object = {}
            results = model(frame)
            print(results.pandas())
            img_result = results.imgs[0]
            df = results.pandas().xyxy[0]
            print(df)
            if df.empty:
                print('DataFrame is empty')
            else:
                b_isTraffic = False
                for index, row in df.iterrows():
                    # 2: car, 3: motorcycle 5: bus, 7:truck
                    if row.iloc[5] == 2 or row.iloc[5] == 3 or row.iloc[5] == 5 or row.iloc[5] == 7:
                        b_isTraffic = True
                        if row.confidence > thresh_confidence:
                            # initialization
                            b_isNewObject = True
                            distance_min = 10000
                            # draw rectangle for the detection
                            cv2.rectangle(img_result, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)),
                                          color=(255, 0, 0), thickness=5)
                            x_detect = (row.xmin + row.xmax) * 0.5
                            y_detect = (row.ymin + row.ymax) * 0.5
                            width_detect = row.xmax - row.xmin
                            height_detect = row.ymax - row.ymin

                            # add detected object info.
                            detected_object[id_detect] = Object(id_detect, x=x_detect, y=y_detect, width=width_detect,
                                                                height=height_detect)
                            id_detect = id_detect + 1

                            # for all existing object
                            for obj in i_object.values():
                                r = obj
                                # distance check
                                distance_x = abs(r.x - x_detect)
                                distance_y = abs(r.y - y_detect)
                                r.set_distance_min(distance_x, distance_y)
                                r.check_close_object(x_detect, y_detect, width_detect, height_detect)
                                if distance_x < p_distance_x and distance_y < p_distance_y:
                                    r.update_object(x_detect, y_detect, width_detect, height_detect)
                                    b_isNewObject = False

                            if b_isNewObject and in_position_object_creation(x_detect, y_detect, i_object):
                                i_object[id_t] = Object(id_t, x=x_detect, y=y_detect, width=width_detect,
                                                        height=height_detect)
                                i_object[id_t].set_distance_min(0.0, 0.0)
                                id_t = id_t + 1
                        # ignore low probability object
                        else:
                            cv2.rectangle(img_result, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)),
                                          color=(0, 255, 0), thickness=5)

                # Remove all object if no object is detected
                if not b_isTraffic:
                    for obj in i_object.values():
                        obj.count_no_detection()

            id_delete = []
            for obj in i_object.values():
                if obj.id not in id_delete:
                    for obj2 in i_object.values():
                        if obj.id != obj2.id:
                            if obj.distance(obj2.x, obj2.y) < p_distance:
                                obj2.delete_object(id_delete)
                if obj.line_crossed:
                    cv2.putText(img_result, text='counted', org=(int(obj.x), int(obj.y)),
                                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 0, 0),
                                thickness=5, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(img_result, text='%d' % obj.id, org=(int(obj.x), int(obj.y)),
                                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255),
                                thickness=5, lineType=cv2.LINE_AA)
                    cv2.ellipse(img_result, ((int(obj.x), int(obj.y)), (p_distance_x * 2, p_distance_y * 2), 0),
                                color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)

                if obj.distance_min > 200:
                    obj.count_no_detection()
                obj.reset_distance_min()

                obj.delete_object_counter(id_delete)

            for id_l in id_delete:
                del i_object[id_l]

            id_delete = []

            # Check Crossing
            for obj in i_object.values():
                if obj.check_crossing_border_left(crossing_boarder_x):
                    counter_traffic = counter_traffic + 1
                    cv2.putText(img_result, text='%d' % obj.id, org=(int(r.x), int(r.y)),
                                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255),
                                thickness=5, lineType=cv2.LINE_AA)

            cv2.putText(img_result, text='Traffic to left: %d' % counter_traffic, org=(crossing_boarder_x-100, 700),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(255, 255, 255),
                        thickness=5, lineType=cv2.LINE_AA)

            key = cv2.waitKey(delay)
            if key == ord('q'):
                break
            if key == ord('s'):
                cv2.waitKey(-1)
            # Display
            cv2.imshow(window_name, img_result)
        else:
            break

    # post processing
    print('Traffic to left: %d' % counter_traffic)
    cv2.destroyWindow(window_name)
