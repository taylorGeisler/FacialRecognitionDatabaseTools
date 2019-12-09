import os
import sys
import csv
import numpy as np
from numpy import linalg as LA
import pandas as pd
import shutil
from tensorflow.keras.models import load_model
from PIL import Image

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

class frdt_face_t:
  def __init__(self, features_=np.empty(128, dtype=np.float64), id_=-1, is_classified_=False):
    self.face_features_ = features_
    self.face_id_ = id_
    self.is_classified_ = is_classified_
    self.is_face_ = True

class frdt_person_t:
  def __init__(self, face_ids_=np.empty(0, dtype=np.float64), person_info_=''):
    self.face_ids_ = face_ids_
    self.person_info_ = person_info_
    
  def add_face(self, face_id_):
    self.face_ids_ = np.append(self.face_ids_, face_id_)

class frdt_source_t:
  def __init__(self, source_id_, face_ids_=np.empty(0, dtype=np.float64), is_video_=False, source_info_=''):
    self.source_id_ = source_id_
    self.face_ids_ = face_ids_
    self.is_video_ = is_video_
    self.source_info_ = source_info_
    
  def add_face(self, face_id_):
    self.face_ids_ = np.append(self.face_ids_, face_id_)

class frdt_database_t:
  def __init__(self, db_directory_, facenet_model_=load_model('facenet_keras.h5', compile=False), facenet_model_im_size_=(160, 160)):
    self.db_directory_ = db_directory_
    self.facenet_model_ = facenet_model_
    self.facenet_model_im_size_ = facenet_model_im_size_
    self.faces_ = np.array(0, dtype=object)
    self.sources_ = np.array(0, dtype=object)
    self.people_ = np.array(0, dtype=object)
    self.faces_empty_ = True
    self.sources_empty_ = True
    self.people_empty_ = True
    
  def num_faces(self):
    if self.faces_empty_:
      return 0
    else:
      return np.size(self.faces_)

  def num_people(self):
    if self.people_empty_:
      return 0
    else:
      return np.size(self.people_)
    
  def num_sources(self):
    if self.sources_empty_:
      return 0
    else:
      return np.size(self.sources_)
    
  def add_face(self, face_):
    if self.faces_empty_:
      self.faces_ = np.array([face_])
      self.faces_empty_ = False
    else:
      self.faces_ = np.append(self.faces_, face_)
      
  def compute_face(self, image_file_, face_id_):
    image = Image.open(self.db_directory_ + 'faces/' + os.fsdecode(image_file_))
    image = image.convert('RGB')
    image = image.resize(self.facenet_model_im_size_)
    face_pixels = np.asarray(image)
    embedding = get_embedding(self.facenet_model_, face_pixels)
    print(LA.norm(embedding))
    return frdt_face_t(embedding, face_id_)
  
  def add_faces_dir(self, dir_):
    for file in sorted(os.listdir(dir_)):
      filename = os.fsdecode(file)
      if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        face_source = dir_ + filename
        if self.faces_empty_:
          face_destination = self.db_directory_ + 'faces/' + str(0).zfill(10) + '.jpeg'
          self.add_face(self.compute_face(file,0))
        else:
          face_destination = self.db_directory_ + 'faces/' + str(self.num_faces()).zfill(10) + '.jpeg'
          self.add_face(self.compute_face(file,self.num_faces()))
        shutil.copyfile(face_source, face_destination)

  def exclude_face(self, face_id_):
    self.faces_[face_id_].is_face = False
    excluded_faces_filename = self.db_directory_ + 'excluded_faces.npy'
    excluded_faces = np.load(self.db_directory_+'excluded_faces.npy')
    excluded_faces = np.append(excluded_faces, face_id_)
    np.save(self.db_directory_+'excluded_faces.npy', excluded_faces)
    
  def exclude_loaded_face(self, face_id_):
    self.faces_[face_id_].is_face = False
    
  def get_face(self, face_id_):
    return self.faces_[face_id_]
    
  def get_source(self, source_id_):
    return self.sources_[source_id_]
    
  def get_person(self, person_id_):
    return self.people_[person_id_]
    
  def add_source(self, source_):
    if self.sources_empty_:
      self.sources_ = np.array([source_])
      self.sources_empty_ = False
    else:
      self.sources_ = np.append(self.sources_, source_)
    
  def create_source(self):
    source_contents = np.empty(0, dtype=int)
    if self.sources_empty_:
      np.save(self.db_directory_+'sources/'+str(0).zfill(10)+'.npy', source_contents)
    else:
      np.save(self.db_directory_+'sources/'+str(self.num_sources()).zfill(10)+'.npy', source_contents)
    self.add_source(frdt_source_t(self.num_sources()))
    return self.num_sources()

  def add_face_to_source(self, source_id_, face_id_):
    self.get_source(source_id_).add_face(face_id_)
    source_filename = self.db_directory_+'sources/'+str(source_id_).zfill(10)+'.npy'
    source_contents = np.load(source_filename)
    source_contents = np.append(source_contents, face_id_)
    np.save(self.db_directory_+'sources/'+str(source_id_).zfill(10)+'.npy', source_contents)
    
  def add_person(self, person_):
    if self.people_empty_:
      self.people_ = np.array([person_])
      self.people_empty_ = False
    else:
      self.people_ = np.append(self.people_, person_)

  def create_person(self):
    person_contents = np.empty(0, dtype=int)
    if self.people_empty_:
      np.save(self.db_directory_+'people/'+str(0).zfill(10)+'.npy', person_contents)
    else:
      np.save(self.db_directory_+'people/'+str(self.num_people()).zfill(10)+'.npy', person_contents)
    self.add_person(frdt_source_t(self.num_people()))
    return self.num_people()

  def add_face_to_person(self, person_id_, face_id_):
    self.get_person(person_id_).add_face(face_id_)
    person_filename = self.db_directory_+'people/'+str(person_id_).zfill(10)+'.npy'
    person_contents = np.load(person_filename)
    person_contents = np.append(person_contents, face_id_)
    np.save(self.db_directory_+'people/'+str(person_id_).zfill(10)+'.npy', person_contents)
    
  def load_data(self):
    faces_dir = os.fsencode(self.db_directory_ + 'faces/')
    people_dir = os.fsencode(self.db_directory_ + 'people/')
    sources_dir = os.fsencode(self.db_directory_ + 'sources/')
    excluded_faces_filename = self.db_directory_ + 'excluded_faces.npy'

    # Load faces
    face_id = 0
    for file in sorted(os.listdir(faces_dir)):
      filename = os.fsdecode(file)
      if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        self.add_face(self.compute_face(file,face_id))
        face_id += 1

    # Load people
    for file in sorted(os.listdir(people_dir)):
      person_faces = np.empty(0, dtype=int)
      filename = os.fsdecode(file)
      if filename.endswith('.npy'):
        person_faces = np.load(os.fsdecode(people_dir)+filename)
        self.add_person(frdt_person_t(person_faces))

    # Load sources
    for file in sorted(os.listdir(sources_dir)):
      source_faces = np.empty(0, dtype=int)
      filename = os.fsdecode(file)
      if filename.endswith('.npy'):
        source_faces = np.load(os.fsdecode(sources_dir)+filename)
        self.add_source(frdt_source_t(source_faces))

    # Load Excluded Faces
    excluded_faces = np.load(excluded_faces_filename)
    for face in excluded_faces:
      self.exclude_loaded_face(face)

  def make_new_database_dir(self):
    os.makedirs(self.db_directory_, exist_ok=True)
    os.makedirs(self.db_directory_+'faces/', exist_ok=True)
    os.makedirs(self.db_directory_+'people/', exist_ok=True)
    os.makedirs(self.db_directory_+'sources/',exist_ok=True)
    excluded_faces = np.empty(0, dtype=int)
    np.save(self.db_directory_+'excluded_faces.npy', excluded_faces)

##########################################################################################
# main program
##########################################################################################

db_directory = '/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/2/'
database = frdt_database_t(db_directory)
database.load_data()
# 
# db_directory1 = '/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/1/'
# database1 = frdt_database_t(db_directory1)
# database1.make_new_database_dir()
# 
# faces_dir = '/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/0/faces/'
# database1.add_faces_dir(faces_dir)
# 
# for i in range(10):
#   database.create_source()
#   
# for i in range(9):
#   database.create_person()
#   
# 
# database1.exclude_face(5)
# database1.exclude_face(7)
# 
# database1.add_face_to_source(1,5)
# database1.add_face_to_source(1,7)

# db_directory2 = '/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/3/'
# database2 = frdt_database_t(db_directory2)
# database2.make_new_database_dir()
# 
# for i in range(10):
#   database2.create_source()
# 
# for i in range(10):
#   database2.create_person()
# 
# faces_dir = '/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/0/faces/'
# database2.add_faces_dir(faces_dir)
# 
# for i in range(12):
#   database2.exclude_face(i)
# 
# database2.add_face_to_source(9,5)
# database2.add_face_to_source(1,7)
# 
# database2.add_face_to_person(9,5)
# database2.add_face_to_person(1,7)

print('Number of faces: ' + str(database.num_faces()))
print('Number of people: ' + str(database.num_people()))
print('Number of sources: ' + str(database.num_sources()))