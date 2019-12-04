import os
import sys
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

  def exclude_face(self):
    pass
    
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
    
  def remove_person(self):
    pass
    
  def check_person(self):
    pass
    
  def merge_people(self):
    pass
    
  def load_data(self):
    pass
    
  def save_data(self):
    pass

A = frdt_face_t()
B = frdt_person_t()
C = frdt_source_t()
D = frdt_database_t()
D.add_face(A)
D.add_face(A)
D.add_face(frdt_face_t())
D.add_person(B)
D.add_source(C)


print(D.faces_)
print(D.sources_[0])
print(D.people_[0])
  
print('Hello world!')