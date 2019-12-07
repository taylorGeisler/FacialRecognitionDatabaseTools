import os
import sys
import csv
import numpy as np
import pandas as pd
import shutil

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
  def __init__(self, db_directory_):
    self.db_directory_ = db_directory_
    self.faces_ = np.array(0, dtype=object)
    self.sources_ = np.array(0, dtype=object)
    self.people_ = np.array(0, dtype=object)
    self.faces_empty_ = True
    self.sources_empty_ = True
    self.people_empty_ = True
    
  def num_faces(self):
    return np.size(self.faces_)

  def num_people(self):
    return np.size(self.people_)
    
  def num_sources(self):
    return np.size(self.sources_)
    
  def add_face(self, face_):
    if self.faces_empty_:
      self.faces_ = np.array([face_])
      self.faces_empty_ = False
    else:
      self.faces_ = np.append(self.faces_, face_)
      
  def compute_face(self, image_file_, face_id_):
    return frdt_face_t(np.random.rand(128), face_id_)
  
  def add_faces_dir(self, dir_):
    face_id = self.num_faces()
    for file in os.listdir(dir_):
      filename = os.fsdecode(file)
      if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        face_source = dir_ + filename
        if self.faces_empty_:
          face_destination = self.db_directory_ + 'faces/' + str(0).zfill(10) + '.jpeg'
        else:
          face_destination = self.db_directory_ + 'faces/' + str(face_id).zfill(10) + '.jpeg'
          face_id += 1
        shutil.copyfile(face_source, face_destination) 
        self.add_face(self.compute_face(file,face_id))

  def exclude_face(self, face_id_):
    self.faces_[face_id_].is_face = False
    excluded_faces_filename = self.db_directory_ + 'excluded_faces.csv'
    if (os.stat(excluded_faces_filename).st_size == 0):
      excl_file = open(self.db_directory_+'excluded_faces.csv' ,'a')
      excl_file.write(str(face_id_))
      excl_file.close()
    else:
      df = pd.read_csv(excluded_faces_filename, sep=',', header=None)
      df[len(df.columns)] = face_id_
      print(df.values)
      df.to_csv(excluded_faces_filename)
    
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
    if self.sources_empty_:
      open(self.db_directory_+'sources/'+str(0).zfill(10)+'.csv' ,'a').close()
    else:
      open(self.db_directory_+'sources/'+str(self.num_sources()).zfill(10)+'.csv' ,'a').close()
    self.add_source(frdt_source_t(self.num_sources()))
    return self.num_sources()

  def add_face_to_source(self, source_id_, face_id_):
    self.get_source(source_id_).add_face(face_id_)
    source_filename = self.db_directory_+'sources/'+str(source_id_).zfill(10)+'.csv'
    df = pd.read_csv(source_filename, sep=',', header=None)
    df[len(df.columns)] = face_id_
    df.to_csv(source_filename)
    
  def add_person(self, person_):
    if self.people_empty_:
      self.people_ = np.array([person_])
      self.people_empty_ = False
    else:
      self.people_ = np.append(self.people_, person_)

  def create_person(self):
    if self.people_empty_:
      open(self.db_directory_+'people/'+str(0).zfill(10)+'.csv' ,'a').close()
    else:
      open(self.db_directory_+'people/'+str(self.num_people()).zfill(10)+'.csv' ,'a').close()
    self.add_person(frdt_source_t(self.num_people()))
    return self.num_people()

  def add_face_to_person(self, person_id_, face_id_):
    self.get_person(person_id_).add_face(face_id_)
    person_filename = self.db_directory_+'people/'+str(source_id_).zfill(10)+'.csv'
    df = pd.read_csv(person_filename, sep=',', header=None)
    df[len(df.columns)] = face_id_
    df.to_csv(person_filename)
    
  def load_data(self):
    faces_dir = os.fsencode(self.db_directory_ + '/faces/')
    people_dir = os.fsencode(self.db_directory_ + '/people/')
    sources_dir = os.fsencode(self.db_directory_ + '/sources/')
    excluded_faces_filename = self.db_directory_ + 'excluded_faces.csv'

    # Load faces
    face_id = 0
    for file in os.listdir(faces_dir):
      filename = os.fsdecode(file)
      if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        self.add_face(self.compute_face(file,face_id))
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
        self.add_person(frdt_person_t(person_faces))
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
              source_faces = np.append(source_faces,int(c))
        self.add_source(frdt_source_t(source_faces))
        source_id += 1

    # Load Excluded Faces
    with open(excluded_faces_filename,'r') as f:
      reader = csv.reader(f, delimiter=',')
      for row in reader:
        for c in row:
          self.exclude_loaded_face(int(c))

  def make_new_database_dir(self):
    os.makedirs(self.db_directory_, exist_ok=True)
    os.makedirs(self.db_directory_+'faces/', exist_ok=True)
    os.makedirs(self.db_directory_+'people/', exist_ok=True)
    os.makedirs(self.db_directory_+'sources/',exist_ok=True)
    open(self.db_directory_+'excluded_faces.csv' ,'a').close()
    
###

db_directory = '/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/0/'
database = frdt_database_t(db_directory)
database.load_data()

db_directory1 = '/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/1/'
database1 = frdt_database_t(db_directory1)
database1.make_new_database_dir()

faces_dir = '/Users/taylor/Google Drive/Developer/computer_vision/frdt_databases/0/faces/'
database1.add_faces_dir(faces_dir)

for i in range(10):
  database1.create_source()
  
for i in range(9):
  database1.create_person()
  

database1.exclude_face(5)
database1.exclude_face(7)

database1.add_face_to_source(1,5)
database1.add_face_to_source(1,7)
  

print('Number of faces: ' + str(database.num_faces()))
print('Number of people: ' + str(database.num_people()))
print('Number of sources: ' + str(database.num_sources()))