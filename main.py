import os
import sys
import csv
import numpy as np
import pandas as pd

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

class frdt_source_t:
  def __init__(self, source_id_=-1, image_count_=-1, is_video_=False, source_info_=''):
    self.source_id_ = source_id_
    self.image_count_ = image_count_
    self.is_video_ = is_video_
    self.source_info_ = source_info_

class frdt_database_t:
  def __init__(self):
    self.faces_ = np.array(0, dtype=object)
    self.sources_ = np.array(0, dtype=object)
    self.people_ = np.array(0, dtype=object)
    self.faces_empty_ = True
    self.sources_empty_ = True
    self.people_empty_ = True
    
  def add_face(self, face_):
    if self.faces_empty_:
      self.faces_ = np.array([face_])
      self.faces_empty_ = False
    else:
      self.faces_ = np.append(self.faces_, face_)

  def exclude_face(self, face_):
    face_.is_face = False
    
  def verify_faces(self):
    pass
    
  def add_source(self, source_):
    if self.sources_empty_:
      self.sources_ = np.array([source_])
      self.sources_empty_ = False
    else:
      self.sources_ = np.append(self.sources_, source_)

  def remove_source(self):
    pass
    
  def add_person(self, person_):
    if self.people_empty_:
      self.people_ = np.array([person_])
      self.people_empty_ = False
    else:
      self.people_ = np.append(self.people_, person_)
    
  def remove_person(self, person_):
    pass
    
  def check_person(self):
    pass
    
  def merge_people(self):
    pass
    
  def compute_face(self, image_file_):
    return np.random.rand(128)
    
  def load_data(self, db_directory):
    faces_dir = os.fsencode(db_directory + '/faces/')
    people_dir = os.fsencode(db_directory + '/people/')
    sources_dir = os.fsencode(db_directory + '/sources/')
    excluded_faces_filename = db_directory + 'excluded_faces.csv'

    # Load faces
    face_id = 0
    for file in os.listdir(faces_dir):
      filename = os.fsdecode(file)
      if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # add_face(frdt_face_t(compute_face_features(file),face_id))
        face_id += 1

    # Load people
    person_id = 0
    for file in os.listdir(people_dir):
      person_faces = np.empty(0, dtype=int)
      filename = os.fsdecode(file)
      if filename.endswith('.csv'):
        with open(os.fsdecode(people_dir)+filename,'r') as f:
          reader = csv.reader(f, delimiter=',')
          for row in reader:
            for c in row:
              person_faces = np.append(person_faces,int(c))
        #add_person()
        person_id += 1

    # Load sources
    source_id = 0
    for file in os.listdir(sources_dir):
      source_faces = np.empty(0, dtype=int)
      filename = os.fsdecode(file)
      if filename.endswith('.csv'):
        with open(os.fsdecode(sources_dir)+filename,'r') as f:
          reader = csv.reader(f, delimiter=',')
          for row in reader:
            for c in row:
              person_faces = np.append(person_faces,int(c))
        #add_source()
        source_id += 1

    # Load Excluded Faces
    excluded_faces = np.empty(0, dtype=int)
    with open(excluded_faces_filename,'r') as f:
      reader = csv.reader(f, delimiter=',')
      for row in reader:
        for c in row:
          excluded_faces = np.append(excluded_faces,int(c))

  def save_data(self):
    pass
    
database = frdt_database_t()
database.load_data('/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/0/')
  
print('Hello world!')