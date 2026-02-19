import cv2
import torch
import torchvision.transforms as transforms

preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),  # transforms to python format
        transforms.Resize((224, 224)),  # resizes it to fit in the model
        transforms.ToTensor(),  # converts into a tensor
    ]
)


path2vid = "/Volumes/TIZIANO/stimuli/Project1917_movie_part3_24Hz.mp4"
reader = cv2.VideoCapture(path2vid)
tensor_list = []
while True:
    ret, frame = reader.read()
    if not ret:
        break
        # if not ret:
    frame_rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )  # converts to bgr to rgb color codes
    input_tensor = preprocess(
        frame_rgb
    )  # .unsqueeze(0) # unsqueeze adds the batch size in front of the img
    tensor_list.append(input_tensor)
    # while True:

big_tensor = torch.stack(tensor_list, dim=3)
avg = big_tensor.mean(dim=(1, 2, 3))
stdv = big_tensor.std(dim=(1, 2, 3))
print("average:", avg)
print("std", stdv)
