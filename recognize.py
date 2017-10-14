import dlib
from skimage import io
from scipy.spatial import distance 

#  может стоит ещё уменьшить
ENOUGH_DISTANCE = 0.55
#  это вынести в settings
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(
    'dlib_face_recognition_resnet_model_v1.dat')
    
def findFace(image, detector, sp):
    dets = detector(image, 1)
    for k,d in enumerate(dets):
        shape = sp(image, d)
    
    return shape


#  это через comprehensions
#  имхо читать невозможно
#  выберите сами 
def recognize2(userFace, dbFaces):
      
    userImage = io.imread(userFace)
    detector = dlib.get_frontal_face_detector()
    shape = findFace(userImage, detector, sp)
    userFaceDescriptor = facerec.compute_face_descriptor(userImage, shape)
    
    dbImages = [io.imread(dbFace) for dbFace in dbFaces]
    dbFacesDescriptors = [facerec.compute_face_descriptor(
        dbImage, 
        findFace(dbImage, detector, sp))
        for dbImage in dbImages]
    distanceToFaces = [distance.euclidean(userFaceDescriptor, dbFaceDescriptor)
            for dbFaceDescriptor in dbFacesDescriptors]
    
    return min(distanceToFaces) < ENOUGH_DISTANCE


#  userFace - имя файла c фото пользователя
#  dbFaces - список с именами файлов фото из базы данных
def recognize(userFace, dbFaces):
    
    userImage = io.imread(userFace)
    detector = dlib.get_frontal_face_detector()
    shape = findFace(userImage, detector, sp)
    userFaceDescriptor = facerec.compute_face_descriptor(userImage, shape)
    
    distanceToDbFaces = []
    for dbFace in dbFaces:
        dbImage = io.imread(dbFace)
        shape = findFace(dbImage, detector, sp)
        dbFaceDescriptor = facerec.compute_face_descriptor(dbImage, shape)
        dist = distance.euclidean(userFaceDescriptor, dbFaceDescriptor)
        distanceToDbFaces.append(dist)
    return min(distanceToDbFaces) < ENOUGH_DISTANCE



