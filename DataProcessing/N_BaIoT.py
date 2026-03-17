from DataLoader.utils import *
from zipfile import ZipFile, is_zipfile
from tqdm.auto import tqdm
import time 
SEED = 42

class N_BaIoT():
  
  def __init__(base_self, seed = SEED, print_able = True, save_csv = True) -> None:
    np.random.seed(seed)
    np.set_printoptions(suppress=True)
    base_self.__PRINT_ABLE = print_able
    base_self.__SAVE_CSV = save_csv
    base_self.__data_df = pd.DataFrame()
    base_self.__device_names = [
      "Danmini_Doorbell",
      "Ecobee_Thermostat",
      "Ennio_Doorbell",
      "Philips_B120N10_Baby_Monitor",
      "Provision_PT_737E_Security_Camera",
      "Provision_PT_838_Security_Camera",
      "Samsung_SNH_1011_N_Webcam",
      "SimpleHome_XCS7_1002_WHT_Security_Camera",
      "SimpleHome_XCS7_1003_WHT_Security_Camera"
    ]
    base_self.__target_variable = "Label"
    base_self.__label_fts_names = [base_self.__target_variable]

    base_self.__fts_names = [
      'MI_dir_L5_weight', 'MI_dir_L5_mean', 'MI_dir_L5_variance', 'MI_dir_L3_weight', 'MI_dir_L3_mean', 'MI_dir_L3_variance',
      'MI_dir_L1_weight', 'MI_dir_L1_mean', 'MI_dir_L1_variance', 'MI_dir_L0.1_weight', 'MI_dir_L0.1_mean', 'MI_dir_L0.1_variance',
      'MI_dir_L0.01_weight', 'MI_dir_L0.01_mean', 'MI_dir_L0.01_variance', 'H_L5_weight', 'H_L5_mean', 'H_L5_variance', 'H_L3_weight',
      'H_L3_mean', 'H_L3_variance', 'H_L1_weight', 'H_L1_mean', 'H_L1_variance', 'H_L0.1_weight', 'H_L0.1_mean', 'H_L0.1_variance',
      'H_L0.01_weight', 'H_L0.01_mean', 'H_L0.01_variance', 'HH_L5_weight', 'HH_L5_mean', 'HH_L5_std', 'HH_L5_magnitude', 'HH_L5_radius',
      'HH_L5_covariance', 'HH_L5_pcc', 'HH_L3_weight', 'HH_L3_mean', 'HH_L3_std', 'HH_L3_magnitude', 'HH_L3_radius', 'HH_L3_covariance',
      'HH_L3_pcc', 'HH_L1_weight', 'HH_L1_mean', 'HH_L1_std', 'HH_L1_magnitude', 'HH_L1_radius', 'HH_L1_covariance', 'HH_L1_pcc',
      'HH_L0.1_weight', 'HH_L0.1_mean', 'HH_L0.1_std', 'HH_L0.1_magnitude', 'HH_L0.1_radius', 'HH_L0.1_covariance', 'HH_L0.1_pcc',
      'HH_L0.01_weight', 'HH_L0.01_mean', 'HH_L0.01_std', 'HH_L0.01_magnitude', 'HH_L0.01_radius', 'HH_L0.01_covariance', 'HH_L0.01_pcc',
      'HH_jit_L5_weight', 'HH_jit_L5_mean', 'HH_jit_L5_variance', 'HH_jit_L3_weight', 'HH_jit_L3_mean', 'HH_jit_L3_variance',
      'HH_jit_L1_weight', 'HH_jit_L1_mean', 'HH_jit_L1_variance', 'HH_jit_L0.1_weight', 'HH_jit_L0.1_mean', 'HH_jit_L0.1_variance',
      'HH_jit_L0.01_weight', 'HH_jit_L0.01_mean', 'HH_jit_L0.01_variance', 'HpHp_L5_weight', 'HpHp_L5_mean', 'HpHp_L5_std',
      'HpHp_L5_magnitude', 'HpHp_L5_radius', 'HpHp_L5_covariance', 'HpHp_L5_pcc', 'HpHp_L3_weight', 'HpHp_L3_mean', 'HpHp_L3_std',
      'HpHp_L3_magnitude', 'HpHp_L3_radius', 'HpHp_L3_covariance', 'HpHp_L3_pcc', 'HpHp_L1_weight', 'HpHp_L1_mean', 'HpHp_L1_std',
      'HpHp_L1_magnitude', 'HpHp_L1_radius', 'HpHp_L1_covariance', 'HpHp_L1_pcc', 'HpHp_L0.1_weight', 'HpHp_L0.1_mean', 'HpHp_L0.1_std',
      'HpHp_L0.1_magnitude', 'HpHp_L0.1_radius', 'HpHp_L0.1_covariance', 'HpHp_L0.1_pcc', 'HpHp_L0.01_weight', 'HpHp_L0.01_mean',
      'HpHp_L0.01_std', 'HpHp_L0.01_magnitude', 'HpHp_L0.01_radius', 'HpHp_L0.01_covariance', 'HpHp_L0.01_pcc'
    ]


      
    base_self.__real_cnt = {
            'benign': 555932,
            'gafgyt.combo': 515156,
            'gafgyt.junk': 261789,
            'gafgyt.scan': 255111,
            'gafgyt.tcp': 859850,
            'gafgyt.udp': 946366,
            'mirai.ack': 643821,
            'mirai.scan': 537979,
            'mirai.syn': 733299,
            'mirai.udp': 1229999,
            'mirai.udpplain': 523304
    }  # Actual number of samples in the csv file of each class
    base_self.__label_map = {
    } # Map the actual label in the csv file with the label in the paper

    base_self.__category_map = {
            'benign': 0,
            'gafgyt.combo': 1,
            'gafgyt.junk': 1,
            'gafgyt.scan': 1,
            'gafgyt.tcp': 1,
            'gafgyt.udp': 1,
            'mirai.ack': 2,
            'mirai.scan': 2,
            'mirai.syn': 2,
            'mirai.udp': 2,
            'mirai.udpplain': 2
    } # Group the actual label in the csv file with the coresponding category in the paper

    base_self.__binary_map = {
            'benign': 0,
            'gafgyt.combo': 1,
            'gafgyt.junk': 1,
            'gafgyt.scan': 1,
            'gafgyt.tcp': 1,
            'gafgyt.udp': 1,
            'mirai.ack': 1,
            'mirai.scan': 1,
            'mirai.syn': 1,
            'mirai.udp': 1,
            'mirai.udpplain': 1
    }
    base_self.__label_true_name = [
            'benign',
            'gafgyt.combo',
            'gafgyt.junk',
            'gafgyt.scan',
            'gafgyt.tcp',
            'gafgyt.udp',
            'mirai.ack',
            'mirai.scan',
            'mirai.syn',
            'mirai.udp',
            'mirai.udpplain'
    ] # Map the actual label in the csv file with the label in the paper

    base_self.__label_drop = [] # List the label be dropped
    base_self.__label_cnt = {} # Actual number of samples loaded by function
    base_self.__error_cnt = 0
    base_self.__set_metadata()
    base_self.Show_basic_metadata()
    base_self.__fixLabel()

  def __set_metadata(base_self) -> None:
    base_self.__ds_name = "N_BaIoT"
    base_self.__ds_size = None
    base_self.__ds_fts = base_self.__fts_names
    base_self.__ds_label = base_self.__label_true_name
    base_self.__ds_paper_link = "https://research.unsw.edu.au/projects/bot-iot-dataset"
    base_self.__ds_link = "https://drive.usercontent.google.com/download?id=148OllYJE-DoYaj6X7MF8hZJb3UJNJnyX&export=download&authuser=2&confirm=t&uuid=2f86317f-1b0b-47ba-90a9-2db9192a242e&at=APZUnTUOIWpWMwN9TT-RkcvYaloi%3A1718122126887"
    base_self.__csv_link = None
    # base_self._ds_name = ""


  def __print(base_self, str) -> None:
    if base_self.__PRINT_ABLE:
      print(str)

  def __add_mode_features(base_self, dataset, FLAG_GENERATING: bool = False) -> pd.DataFrame:
    pass
  
  def __fixLabel(base_self):
    for x in base_self.__label_map:
      y = base_self.__label_map[x]
      if y not in base_self.__real_cnt:
        print('label map ' + x)
        print('real_cnt ' + y)
        base_self.__real_cnt[y] = 0
        base_self.__real_cnt[y] += base_self.__real_cnt[x]
        base_self.__real_cnt.pop(x)
    base_self.__print("True count:")
    base_self.__print(base_self.__real_cnt)

  def __reDefineLabel(base_self):
    base_self.__data_df.rename(columns = {base_self.__target_variable: 'Default_label'}, inplace = True, errors='ignore')
    base_self.__label_fts_names = ["Default_label" if x == base_self.__target_variable else x for x in base_self.__label_fts_names]
    base_self.__fts_names = ["Default_label" if x == base_self.__target_variable else x for x in base_self.__fts_names]
    base_self.__target_variable='Default_label'
    
    base_self.__data_df['Category_label'] = base_self.__data_df[base_self.__target_variable].apply(lambda x: base_self.__category_map[x] if x in base_self.__category_map else x)
    base_self.__label_fts_names.append('Category_label')
    base_self.__fts_names.append('Category_label')

    base_self.__data_df['Binary_label'] = base_self.__data_df[base_self.__target_variable].apply(lambda x: base_self.__binary_map[x] if x in base_self.__binary_map else x)
    base_self.__label_fts_names.append('Binary_label')
    base_self.__fts_names.append('Binary_label')
    return base_self.__data_df

  def __load_raw_default(base_self, dir_path, limit_cnt:sys.maxsize, frac = None):
    base_self.__label_cnt = {}
    base_self.__Null_cnt = 0

    tt_time = time.time()
    df_ans = pd.DataFrame()
    for root, _, files in os.walk(dir_path):
        for file in files:
            base_self.__print("Begin file " + file)
            if not file.endswith(".csv"):
                continue
            # if not file.startswith('5.'):
            #   continue
            list_ss = []
            time_file = time.time()
            for chunk in pd.read_csv(os.path.join(root,file), index_col=None, names=base_self.__fts_names, header=0, chunksize=10000, low_memory=False):
                # # This command is only for CICmMalMem2022
                # chunk[base_self.__target_variable] = ['-'.join(value.split('-')[:2]) for value in chunk[base_self.__target_variable]]
                
                # # This command is only for UNSWNB15
                # chunk.fillna('Normal', inplace=True)

                # This command is only for N_BaIoT
                chunk[base_self.__target_variable] = file[2:-4]
                dfse = chunk[base_self.__target_variable].value_counts()
                
                for x in dfse.index:
                    if x in base_self.__label_drop:
                      continue

                    sub_set = chunk[chunk[base_self.__target_variable] == x]
                    x_cnt = float(dfse[x])

                    if x in base_self.__label_map:
                      x = base_self.__label_map[x]
                    if x not in base_self.__label_cnt:
                        base_self.__label_cnt[x] = 0
                    if limit_cnt == base_self.__label_cnt[x] :
                        continue

                    max_cnt_chunk = min(int(limit_cnt * (x_cnt / base_self.__real_cnt[x]) + 1), sub_set.shape[0])
                    if frac != None:
                        max_cnt_chunk = min(int(frac * x_cnt + 1), sub_set.shape[0])
                    max_cnt_chunk = min(max_cnt_chunk, limit_cnt - base_self.__label_cnt[x])
                    sub_set = sub_set.sample(n=max_cnt_chunk,replace = False, random_state = SEED)
                    sub_set[base_self.__target_variable] = sub_set[base_self.__target_variable].apply(lambda y: base_self.__label_map[y] if y in base_self.__label_map else y)
                    list_ss.append(sub_set)
                    base_self.__label_cnt[x] += sub_set.shape[0]
            df_ans = CustomMerger().fit_transform([df_ans] + list_ss)
        
            print("Update label:")
            print(base_self.__label_cnt)
            print("Time load:", time.time() - time_file)
            print(f"========================== Finish {file} =================================")

    
    print("Total time load:", time.time() - tt_time)
    base_self.__data_df = df_ans
          
  #=========================================================================================================================================

  def __download(base_self, url, filename):
    import functools
    import pathlib
    import shutil
    import requests
    
    r = requests.get(url, stream=True, allow_redirects=True, verify = False)
    if r.status_code != 200:
      r.raise_for_status()  # Will only raise for 4xx codes, so...
      raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))
    print("Start download file - Total file_size:", file_size)
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
      with path.open("wb") as f:
        shutil.copyfileobj(r_raw, f)

    return path  
  
  #=========================================================================================================================================

  def Add_more_fts(base_self):
    #Add more features
    base_self.__print("=================================== Add more features ===================================")
    base_self.__data_df = base_self._add_mode_features(base_self._data_df)
    base_self.__data_df.dropna(inplace =True)
    base_self.__data_df.drop(columns=['1'],inplace =True)
    base_self.__print("=================================== Done Add more features ===================================")
    return


  #=========================================================================================================================================
  
  def DownLoad_Data(base_self, datadir = os.getcwd(),  load_type="raw"):
    print(f"Attention !!! Your chosen dataset will take {base_self.__ds_size} MB in local storage. Use Ctrl+C to abort before the process start.")
    time.sleep(7)     
    if load_type=="preload":
      datapath = os.path.join(datadir, (base_self.__ds_name + "_dataloader.csv"))
      data_url = base_self.__csv_link
      if data_url is None:
        print("Not supported!!!")
        return
      if os.path.exists(datapath) == True:
        print("Data already!!! No need to download.")
        return
      else:
        print("=================================== File Data not found!!! Start downloading ===================================")
        print("File saved at:", base_self.__download(data_url, datapath))
        print("================================ End download data ================================")
        return 
    
    if load_type=="raw":
      data_dir = os.path.join(datadir,  base_self.__ds_name)
      data_file = os.path.join(datadir,  (base_self.__ds_name + ".zip"))
      data_url = base_self.__ds_link
      if os.path.exists(data_dir) == True:
        print("Data already!!! No need to download.")
        return
      else:
        print("=================================== Folder Data not found!!! Start downloading ===================================")
        if os.path.exists(data_file) == False:
          print("=================================== File Data Zip not found!!! Start downloading ===================================")
          print("File saved at:", base_self.__download(data_url, data_file))
          print("================================ End download data ================================")
        if is_zipfile(data_file):
          print("=================================== Unzipping Data!!!===================================")
          os.makedirs(data_dir, exist_ok=True)
          with ZipFile(data_file,"r") as zip_ref:
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
              zip_ref.extract(member=file, path=data_dir)
          print("File saved at:", datadir)
          print("=================================== End download data ===================================")
        else:
          print("=================================== Zip file not valid!!! ===================================")
      return 

  #=========================================================================================================================================

  def Load_Data(base_self, datadir = os.getcwd(), device_name="all", load_type="raw", limit_cnt=sys.maxsize, frac = None):
    if load_type=="preload":
      datapath = os.path.join(datadir, (base_self.__ds_name + "_dataloader.csv"))
      if os.path.exists(datapath) == True:
        print("================================ Start load data ================================")
        base_self.__data_df =  pd.read_csv(datapath, index_col=None, header=0)
        base_self.__reDefineLabel()
        print("================================ Data loaded ================================")
        return
      else:
        print("=================================== File Data not found!!! ===================================")
        return 
    
    if load_type=="raw":
      datapath = os.path.join(datadir, base_self.__ds_name)
      if os.path.exists(datapath) == True:
        print("================================ Start load data ================================")
        if device_name == "all":
          print("You choose all devices.")
        else:
          print("You choose devce:", device_name)
          datapath = os.path.join(datapath, device_name)
        base_self.__load_raw_default(datapath, limit_cnt, frac)
        base_self.__reDefineLabel()
        print("================================ Data loaded ================================")
        return
      else:
        print("=================================== Folder Data not found!!! Please download first ===================================")
      return

  #=========================================================================================================================================
  
  def Encoder_Data(base_self, type_encoder='LabelEncoder'):
    if type_encoder == 'LabelEncoder':
        encoder = LabelEncoder()
    elif type_encoder == 'OneHotEncoder':
        encoder = OneHotEncoder()
    elif type_encoder == 'CustomEncoder':
        encoder = CustomEncoder()
    else:
        raise ValueError('Invalid encoder type. Select: LabelEncoder, OneHotEncoder, CustomEncoder')

    df = base_self.__data_df
    base_self.__print(f"Use {type_encoder} to encode features")
    object_columns = df.select_dtypes(include=['object']).columns
    print(f"List of encoded features: {list(object_columns)}")
    for col in object_columns:
        if col in base_self.__label_fts_names:
            continue
        df[col] = df[col].astype("string")
        df[col] = encoder.fit_transform(df[col])
    base_self.__data_df = df
     
  #=========================================================================================================================================
  
  def Scaler_Data(base_self, type_scaler='QuantileTransformer'):
    if type_scaler == 'StandardScaler':
        scaler = StandardScaler()
    elif type_scaler == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif type_scaler == 'QuantileTransformer':
        scaler = QuantileTransformer()
    elif type_scaler == 'CustomScaler':
        scaler = CustomScaler()
    else:
        raise ValueError('Invalid scaler type. Select: StandardScaler, MinMaxScaler, QuantileTransformer, CustomScaler')
    
    df = base_self.__data_df
    base_self.__print(f"Use {type_scaler} to standardize features")
    for col in df.columns:
        if col in base_self.__label_fts_names:
            continue

        data = df[[col]]
        df[col] = scaler.fit_transform(data).reshape(-1, 1)
    base_self.__data_df = df
       
  #=========================================================================================================================================
  
  def Clean_Data(base_self):
    df = base_self.__data_df
    base_self.__print(f"Start cleanning data for {base_self.__ds_name}")
    X = df.drop(base_self.__label_fts_names, axis=1)

    # Remove zero features (columns)
    base_self.__print("Remove zero features (columns)")
    zero_cols = X.columns[X.isna().all()]
    if len(zero_cols) > 0:
      base_self.__print(f"Zero features (columns) to remove: {list(zero_cols)}")
      X = X.drop(zero_cols, axis=1)

    # Remove duplicated features (columns)  
    base_self.__print("Remove duplicated features (columns) ")  
    dup_cols = X.columns[X.columns.duplicated()]
    if len(dup_cols) > 0:
      base_self.__print(f"Duplicated features (columns) to remove: {list(dup_cols)}")
      X = X.drop(dup_cols, axis=1)

    # Remove constant features (columns)
    base_self.__print("Remove constant features (columns)")
    constant_cols = [col for col in X.columns if X[col].nunique() == 1]
    if len(constant_cols) > 0:
      base_self.__print(f"Constant features (columns) to remove: {list(constant_cols)}")
      X = X.drop(constant_cols, axis=1)

    # Concatenate the target variable and the reduced features DataFrame
    y = df[base_self.__label_fts_names]
    y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    df = pd.concat([X, y], axis=0)
  

    base_self.__print("Remove all null, nan, inf values (rows)")
    df = df.replace([np.inf, -np.inf], np.NaN)
    df = df.dropna(axis='index', how='any')
    # Remove duplicated samples (rows)
    base_self.__print("Remove duplicated samples (rows)")
    df = df.drop_duplicates(df.drop_duplicates(subset=df.columns, keep='first'))
    base_self.__data_df = df
    base_self.__fts_names = base_self.__data_df.columns

  #=========================================================================================================================================

  def Feature_selection(base_self, type_select='SelectKBest', no_fts=None):

    if no_fts == 'all' or no_fts >= len(base_self.__fts_names):
        print ("You select all features or larger. Ignoring")
        return

    if type_select =='SelectKBest':
      selector = SelectKBest(score_func=f_classif, k=no_fts)
    else:
        raise ValueError('Invalid feature selection type. Select one of the types: SelectKBest')
        
    df = base_self.__data_df
    print ("Use SelectKBest to select the",  no_fts, "best features")
    X = df.drop(columns=base_self.__label_fts_names, axis='columns')
    y = df[base_self.__target_variable]
    selected_features = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)     
    seleted_fts_names = list(X.columns[selected_indices]) + base_self.__label_fts_names
    print(f"Selected {no_fts} Features: {seleted_fts_names}")
    base_self.__data_df = base_self.__data_df[seleted_fts_names]
    base_self.__fts_names = seleted_fts_names

  #=========================================================================================================================================
  
  def Preprocess_Data(base_self, type_encoder='LabelEncoder', type_scaler='QuantileTransformer' , type_select='SelectKBest', num_fts='all'):
    print("=============================== Preprocess Data ==================================")
    # df = base_self.__data_df
    # # df = df.drop(columns=[], axis=1)
    # base_self.__data_df = df
    base_self.Clean_Data()
    base_self.Encoder_Data(type_encoder=type_encoder)
    base_self.Scaler_Data(type_scaler=type_scaler)
    base_self.Feature_selection(type_select=type_select, no_fts=num_fts)
    print("==================================================================================")

  #=========================================================================================================================================

  def Train_test_split(base_self, testsize=0.2, target_variable=None , type_classification='binary'):
    if target_variable is None or target_variable not in base_self.__label_fts_names:
      raise ValueError(f"Invalid target variable: {base_self.__label_fts_names}")
    print("============================== Begin Split File ==================================")
    df = base_self.__data_df
    print("============================== Dataframe be like =================================")
    print('\n' + tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql'))

    df_X = df.drop(columns = base_self.__label_fts_names, axis=1)
    df_y = df[target_variable]

    if type_classification == 'binary':
      df_y = df_y.apply(lambda x: 0 if x == "normal" else 1)
    elif type_classification == 'multiclass':
      encoder =  LabelEncoder()
      df_y = encoder.fit_transform(df_y)
      print("Label Encoder classes:", encoder.classes_)
    else:
      raise ValueError('Invalid type classification: Must ''binary'' or ''multiclass''. ')

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y,    
                                                        test_size=testsize, random_state=SEED)

    print("Training data shape:",X_train.shape, y_train.shape)
    print("Testing data shape:",X_test.shape, y_test.shape)

    print("Label Train count:")
    unique= np.bincount(y_train)
    print(np.asarray((unique)))
    print("Label Test count:")
    unique= np.bincount(y_test)
    print(np.asarray((unique)))
    print("=============================== Split File End ===================================")
    return X_train, X_test, y_train, y_test

  #=========================================================================================================================================

  def Show_basic_metadata(base_self):
    print("============================ Show dataset metadata ===============================")
    print("Dataset name:", base_self.__ds_name)
    print("Original public at paper:", base_self.__ds_paper_link)
    print("Dataset link:", base_self.__ds_link)
    print("Total size on disk (MB):", base_self.__ds_size)
    print("Total ",len(base_self.__ds_fts), "features: ")
    print(base_self.__ds_fts)
    print("Total ",len(base_self.__ds_label), "classes: ")
    print(base_self.__ds_label)

  def Show_basic_analysis(base_self):
    print("===================== Show basic analysis of data frame ==========================")
    print("============================= Dataframe be like ==================================")
    print('\n' + tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql'))
    print("================================== Data info =====================================")
    print(base_self.__data_df.info())
    print("============================== Label distribution ================================")
    print(base_self.__data_df[base_self.__target_variable].value_counts())
    
  #=========================================================================================================================================
  
  def To_dataframe(base_self):
    return base_self.__data_df
  
  #=========================================================================================================================================

  def To_csv(base_self, datadir = os.getcwd()):
    file_name = base_self.__ds_name + "_dataloader.csv"
    data_file = os.path.join(datadir, file_name)
    if os.path.exists(data_file) == True:
      print("File is already exists at path:", data_file)
    else:
      base_self.__data_df.to_csv(data_file, index=True)
      print("File saved at:", data_file)
    return