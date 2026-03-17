from DataLoader.utils import *
from tqdm.auto import tqdm
import time 
SEED = 42

class ToNIoT():
  """
  ToNIoT dataset class.
  Parameters:
    seed = int:
      Seed for random function. Default is 42.
    print_able = bool:
      Allow to print description. Default is True.
  """
  def __init__(base_self, seed = SEED, print_able = True, save_csv = True) -> None:
    np.random.seed(seed)
    np.set_printoptions(suppress=True)
    base_self.__PRINT_ABLE = print_able
    base_self.__data_df = pd.DataFrame()
    base_self.__target_variable = 'type'
    base_self.__label_fts_names = ['label', 'type']
    base_self.__fts_names = [
      'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'service',
      'duration', 'src_bytes', 'dst_bytes', 'conn_state', 'missed_bytes',
      'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes', 'dns_query',
      'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA',
      'dns_rejected', 'ssl_version', 'ssl_cipher', 'ssl_resumed',
      'ssl_established', 'ssl_subject', 'ssl_issuer', 'http_trans_depth',
      'http_method', 'http_uri', 'http_referrer', 'http_version',
      'http_request_body_len', 'http_response_body_len', 'http_status_code',
      'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
      'weird_name', 'weird_addl', 'weird_notice', 'label', 'type'
      ]
    base_self.__real_cnt = {
      'normal':     796380,
      'backdoor':   508116,
      'ddos':       6165008,
      'dos':        3375328,
      'injection':  452659, 
      'mitm':       1052,
      'password':   1718568,
      'ransomware': 72805, 
      'scanning':   7140161, 
      'xss':        2108944
    }  # Actual number of samples in the csv file of each class
    base_self.__label_map = {} # Map the actual label in the csv file with the label in the paper

    base_self.__category_map = {
      'normal':    'normal',
      'backdoor':  'backdoor',
      'ddos':      'ddos',
      'dos':       'dos',
      'injection': 'injection',
      'mitm':      'mitm',
      'password':  'password',
      'ransomware':'ransomware',
      'scanning':  'scanning',
      'xss':       'xss'
    } # Group the actual label in the csv file with the coresponding category in the paper

    base_self.__binary_map = {
      'normal':    'Benign',
      'backdoor':  'Malicious',
      'ddos':      'Malicious',
      'dos':       'Malicious',
      'injection': 'Malicious',
      'mitm':      'Malicious',
      'password':  'Malicious',
      'ransomware':'Malicious',
      'scanning':  'Malicious',
      'xss':       'Malicious',
    }
    base_self.__label_true_name = [
        'normal', 'backdoor', 'ddos', 'dos', 'injection', 'mitm','password','ransomware', 'scanning', 'xss'
      ] # Map the actual label in the csv file with the label in the paper

    base_self.__label_drop = [] # List the label be dropped
    base_self.__label_cnt = {} # Actual number of samples loaded by function
    base_self.__data_dir = os.getcwd()
    base_self.__set_metadata()
    base_self.Show_basic_metadata()
    base_self.__fixLabel()

  def __set_metadata(base_self) -> None:
    base_self.__ds_name = "ToNIoT"
    base_self.__ds_size = None
    base_self.__ds_fts = base_self.__fts_names
    base_self.__ds_label = base_self.__label_true_name
    base_self.__ds_paper_link = "https://research.unsw.edu.au/projects/toniot-datasets"
    base_self.__ds_link = "https://drive.usercontent.google.com/download?id=10Nk4Tpo2wcct1596DXE15J6v87Cg4_xz&export=download&authuser=1&confirm=t&uuid=657cc769-d1d3-4988-b2be-3c4fd51000e4&at=APZUnTXCdY2zd-1YQN6VRgN7sOsB:1720492184445"
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
    base_self.__print(f"Start Redefine Label.")
    base_self.__data_df.rename(columns = {base_self.__target_variable: 'Default_label'}, inplace = True, errors='ignore')
    base_self.__label_fts_names = ["Default_label" if x == base_self.__target_variable else x for x in base_self.__label_fts_names]
    base_self.__fts_names = ["Default_label" if x == base_self.__target_variable else x for x in base_self.__fts_names]
    base_self.__target_variable = 'Default_label'
    if 'Category_label' not in base_self.__fts_names:
        base_self.__data_df['Category_label'] = base_self.__data_df[base_self.__target_variable].apply(lambda x: base_self.__category_map[x] if x in base_self.__category_map else x)
        base_self.__label_fts_names.append('Category_label')
        base_self.__fts_names.append('Category_label')
    if 'Binary_label' not in base_self.__fts_names:
        base_self.__data_df['Binary_label'] = base_self.__data_df[base_self.__target_variable].apply(lambda x: base_self.__binary_map[x] if x in base_self.__binary_map else x)
        base_self.__label_fts_names.append('Binary_label')
        base_self.__fts_names.append('Binary_label')
    return base_self.__data_df

  def __load_raw_default(base_self, dir_path, limit_cnt:sys.maxsize, frac = None):
    base_self.__label_cnt = {}

    tt_time = time.time()
    df_ans = pd.DataFrame()
    for root, _, files in os.walk(dir_path):
        for file in files:
            base_self.__print("Begin file " + file)
            if not file.endswith(".csv"):
                continue
            list_ss = []
            time_file = time.time()
            for chunk in pd.read_csv(os.path.join(root,file), index_col=None, names=base_self.__fts_names, header=0, chunksize=10000, low_memory=False):

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
        
            base_self.__print("Update label:")
            base_self.__print(base_self.__label_cnt)
            base_self.__print(f"Time load: {time.time() - time_file}")
            base_self.__print(f"================================ Finish {file} ===================================")
    
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
    zip_file_size = file_size / (1024 * 1024)
    base_self.__print(f"Attention !!! Your chosen dataset will take {zip_file_size:.2f} MB in local storage. Use Ctrl+C to abort before the process start.")
    base_self.__print("Start download file:")
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
      with path.open("wb") as f:
        shutil.copyfileobj(r_raw, f)

    return path  
       
  #=========================================================================================================================================
  
  def __cleanData(base_self, drop_cols=None):
    df = base_self.__data_df
    if drop_cols is not None:
        for col_name in drop_cols: 
            if col_name in base_self.__label_fts_names:
                base_self.__label_fts_names.remove(col_name)
        base_self.__print(f"Drop columns: {drop_cols}")
        df = df.drop(columns=drop_cols, axis=1)
    base_self.__print(f"Start cleanning data for {base_self.__ds_name}")
    X = df.drop(base_self.__label_fts_names, axis=1)

    # Remove zero features (columns)
    base_self.__print(f"Remove zero features (columns).")
    zero_cols = X.columns[X.isna().all()]
    if len(zero_cols) > 0:
      base_self.__print(f"Zero features (columns) to remove: {list(zero_cols)}")
      X = X.drop(zero_cols, axis=1)

    # Remove duplicated features (columns)  
    base_self.__print(f"Remove duplicated features (columns).")  
    dup_cols = X.columns[X.columns.duplicated()]
    if len(dup_cols) > 0:
      base_self.__print(f"Duplicated features (columns) to remove: {list(dup_cols)}")
      X = X.drop(dup_cols, axis=1)

    # Remove constant features (columns)
    base_self.__print(f"Remove constant features (columns).")
    constant_cols = [col for col in X.columns if X[col].nunique() == 1]
    if len(constant_cols) > 0:
      base_self.__print(f"Constant features (columns) to remove: {list(constant_cols)}")
      X = X.drop(constant_cols, axis=1)

    # Concatenate the target variable and the reduced features DataFrame
    y = df[base_self.__label_fts_names]
    y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    df = pd.concat([X, y], axis=1)
  

    base_self.__print(f"Remove all null, nan, inf values (rows).")
    df = df.replace([np.inf, -np.inf], np.NaN)
    df = df.dropna(axis='index', how='any')
    # Remove duplicated samples (rows)
    base_self.__print(f"Remove duplicated samples (rows).")
    df = df.drop_duplicates(subset=X.columns, keep='first')
    base_self.__data_df = df
    base_self.__fts_names = base_self.__data_df.columns

  #=========================================================================================================================================
  
  def __encoderData(base_self, type_encoder='LabelEncoder'):
    if type_encoder == 'LabelEncoder':
        encoder = LabelEncoder()
    elif type_encoder == 'OneHotEncoder':
        encoder = OneHotEncoder()
    elif type_encoder == 'CustomEncoder':
        encoder = CustomEncoder()
    else:
        raise ValueError('Invalid encoder type. Select: LabelEncoder, OneHotEncoder, CustomEncoder!!!')

    df = base_self.__data_df
    base_self.__print(f"Use {type_encoder} to encode features.")
    object_columns = df.select_dtypes(include=['object']).columns
    if len(object_columns) == 0:
        base_self.__print(f"No object columns to encode.")
    else:
        base_self.__print(f"List of encoded features: {list(object_columns)}")
        for col in object_columns:
            if col in base_self.__label_fts_names:
                continue
            df[col] = df[col].astype("string")
            df[col] = encoder.fit_transform(df[col])
    base_self.__data_df = df
     
  #=========================================================================================================================================
  
  def __scalerData(base_self, type_scaler='QuantileTransformer'):
    if type_scaler == 'StandardScaler':
        scaler = StandardScaler()
    elif type_scaler == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif type_scaler == 'QuantileTransformer':
        scaler = QuantileTransformer()
    elif type_scaler == 'CustomScaler':
        scaler = CustomScaler()
    else:
        raise ValueError("Invalid scaler type. Select: StandardScaler, MinMaxScaler, QuantileTransformer, CustomScaler!!!")
    
    df = base_self.__data_df
    base_self.__print(f"Use {type_scaler} to standardize features.")
    for col in df.columns:
        if col in base_self.__label_fts_names:
            continue

        data = df[[col]]
        df[col] = scaler.fit_transform(data).reshape(-1, 1)
    base_self.__data_df = df

  #=========================================================================================================================================

  def __featuresSelection(base_self, type_select='SelectKBest', no_fts=None):

    if no_fts == 'all' or no_fts >= len(base_self.__fts_names):
        base_self.__print(f"You select all features or larger. Ignoring...")
        return

    if type_select =='SelectKBest':
      selector = SelectKBest(score_func=f_classif, k=no_fts)
    else:
        raise ValueError('Invalid feature selection type. Select one of the types: SelectKBest !!!')
        
    df = base_self.__data_df
    base_self.__print(f"Use SelectKBest to select the {no_fts} best features.")
    X = df.drop(columns=base_self.__label_fts_names, axis='columns')
    y = df[base_self.__target_variable]
    selected_features = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)     
    seleted_fts_names = list(X.columns[selected_indices]) + base_self.__label_fts_names
    base_self.__print(f"Selected {no_fts} features: {seleted_fts_names}")
    base_self.__data_df = base_self.__data_df[seleted_fts_names]
    base_self.__fts_names = seleted_fts_names

  # #=========================================================================================================================================

  # def Add_more_fts(base_self):
  #   #Add more features
  #   base_self.__print("=========================== Add more features ===================================")
  #   base_self.__data_df = base_self._add_mode_features(base_self._data_df)
  #   base_self.__data_df.dropna(inplace =True)
  #   base_self.__data_df.drop(columns=['1'],inplace =True)
  #   base_self.__print("========================= Done Add more features ================================")
  #   return


  #=========================================================================================================================================
  
  def DownLoad_Data(base_self, path = None,  load_type="raw"):
    """
    Download the dataset.
    Parameters:
      path = str:
        Path to save the dataset. Default is current working directory.
      load_type = "preload" or "raw":
        Type of dataset to download:There are 2 types of download: "preload" will download a .csv file; "raw" will download the full dataset with a zip file. Default is "raw".
    Returns:
      None
    """
    print("============================= Start download data ================================")
    if path is None:
      datadir = base_self.__data_dir
    else:
      datadir = path
      base_self.__data_dir = datadir
    time.sleep(7)     
    if load_type=="preload":
      datapath = os.path.join(datadir, ("data_" + base_self.__ds_name + ".csv"))
      if os.path.exists(datapath) == True:
        print("Data already!!! No need to download.")
        return
      data_url = base_self.__csv_link
      if data_url is None:
        print("Not supported!!!")
        return
      else:
        print("=================== File Data not found!!! Start downloading =====================")
        datapath = base_self.__download(data_url, datapath)
        print("File saved at:", datapath)
        print("============================== End download data =================================")
        return 
    
    if load_type=="raw":
      zip_file = os.path.join(datadir,  (base_self.__ds_name + ".zip"))
      datadir = os.path.join(datadir,  base_self.__ds_name)
      data_url = base_self.__ds_link
      if os.path.exists(datadir) == True:
        print("Data already!!! No need to download.")
        return
      else:
        print("================= Folder Data not found!!! Start downloading =====================")
        if os.path.exists(zip_file) == False:
          print("================ File Data Zip not found!!! Start downloading ====================")
          print("File Data Zip saved at:", base_self.__download(data_url, zip_file))
          print("============================== End download data =================================")
          print("============================== Unzipping Data!!!==================================")
          
        base_self.__ds_size = ExtractFile(file_path=zip_file, extract_to=datadir)
        # os.makedirs(datadir, exist_ok=True)
        # with ZipFile(zip_file,"r") as zip_ref:
        #   base_self.__ds_size = (sum([info.file_size for info in zip_ref.infolist()])) / (1024 * 1024)
        #   print(f"Attention !!! Your chosen dataset will take {base_self.__ds_size:.2f} MB in local storage. Use Ctrl+C to abort before the process start.")
        #   for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
        #     zip_ref.extract(member=file, path=datadir)
        print("Folder Data saved at:", datadir)
        print("============================== End download data =================================")
      return

  #=========================================================================================================================================

  def Load_Data(base_self, path = None,  load_type="raw",limit_cnt=sys.maxsize, frac = None):
    """
    Load the dataset.
    Parameters:
      path = str:
        Path to the folder containing the dataset. Default is current working directory.
      load_type = "preload" or "raw":
        Load data to dataframe. If using "raw" option, you should specific limit sample of each class by set number to "limit_cnt" because the dataset is very large. Default is "raw".
      limit_cnt = int:
        Maximun sample in each class. Default is sys.maxsize. Only used when load_type="raw"
      frac = float between [0.,1.]:
        Get data by ratio. Default is None. Only used when load_type="raw"
    Returns:
      None
    """
    print("=============================== Start load data ==================================")
    if load_type=="preload":
      if path is None:
        datadir = base_self.__data_dir
        datapath = os.path.join(datadir, ("data_" + base_self.__ds_name + ".csv"))
      else:
        datapath = path
      print("Load Data at:", datapath)
      if os.path.isfile(datapath) == True:
        base_self.__data_df =  pd.read_csv(datapath, index_col=None, header=0, low_memory=False)
        base_self.__reDefineLabel()
        print("================================= Data loaded ====================================")
        return
      else:
        print("============================ File Data not found!!! ==============================")
        return 
    
    if load_type=="raw":
      if path is None:
        datadir = base_self.__data_dir
        datapath = os.path.join(datadir, base_self.__ds_name)
      else:
        datapath = path
      if os.path.exists(datapath) == True:
        base_self.__load_raw_default(datapath, limit_cnt, frac)
        base_self.__reDefineLabel()
        print("================================= Data loaded ====================================")
        return
      else:
        print("=============== Folder Data not found!!! Please download first ===================")
      return

  #=========================================================================================================================================
  
  def Preprocess_Data(base_self, drop_cols=None, type_encoder='LabelEncoder', type_scaler='QuantileTransformer' , type_select='SelectKBest', num_fts='all'):
    """
    Preprocess data by cleaning, encoding, scaling, and selecting features.
    Parameters:
      drop_cols = list:
        List of columns to drop. Default is None.
      type_encoder = "LabelEncoder", "OneHotEncoder" or "CustomEncoder": 
        Type of encoder to encode features. Default is "LabelEncoder".
      type_scaler = "StandardScaler", "MinMaxScaler", "QuantileTransformer" or "CustomScaler":
        Type of scaler to scale features. Default is "QuantileTransformer".
      type_select = SelectKBest:
        Type of feature selection method. Default is "SelectKBest".
      num_fts = int:
        Number of features to keep. Default is 'all'.
    Returns:
      None
    """
    print("=============================== Preprocess Data ==================================")
    print("Start cleaning data...")
    base_self.__cleanData(drop_cols=drop_cols)
    print("Start encoding data...")
    base_self.__encoderData(type_encoder=type_encoder)
    print("Start scaling data...")
    base_self.__scalerData(type_scaler=type_scaler)
    print("Start features selection...")
    base_self.__featuresSelection(type_select=type_select, no_fts=num_fts)
    print("==================================================================================")

  #=========================================================================================================================================

  def Train_test_split(base_self, testsize=0.2, target_variable=None , type_classification='multiclass'):
    """
    Split data into training and testing sets.
    Parameters:
      testsize = float between [0.,1.]:
      Size of the testing set. Default is 0.2.
      target_variable = string:
        Name of column contains the data label. Default is None.
      type_classification = "multiclass" or "binary":
        Type of classification. Default is 'multiclass'.
    Returns:
      X_train = np.array:
        Training features.
      X_test = np.array:
        Testing features.
      y_train = np.array:
        Training labels.
      y_test = np.array:
        Testing labels.
    """
    if target_variable is None or target_variable not in base_self.__label_fts_names:
      raise ValueError(f"Invalid data label!!! Select one of list labels: {base_self.__label_fts_names}")
    print("============================== Begin Split File ==================================")
    df = base_self.__data_df
    base_self.__print("============================== Dataframe be like =================================")
    base_self.__print(f"\n {tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql')}")

    df_X = df.drop(columns = base_self.__label_fts_names, axis=1)
    df_y = df[target_variable]

    if type_classification == 'binary':
      print("Split data for binary classification!!!")
      df_y = df_y.apply(lambda x: 0 if x == "normal" else 1)
    elif type_classification == 'multiclass':
      print("Split data for multiclass classification!!!")
      encoder =  LabelEncoder()
      df_y = encoder.fit_transform(df_y)
      print("Label encoding classes:", encoder.classes_)
    else:
      raise ValueError('Invalid type classification: Must ''binary'' or ''multiclass'' !!!')

    X_train, X_test, y_train, y_test = train_test_split(np.array(df_X), np.array(df_y),    
                                                        test_size=testsize, random_state=SEED)

    base_self.__print(f"Training data shape:{X_train.shape}, {y_train.shape}")
    base_self.__print(f"Testing data shape:{X_test.shape}, {y_test.shape}")

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
    """
    Show basic metadata of the dataset.
    Parameters:
      None
    Returns:
      None
    """
    print("============================ Show dataset metadata ===============================")
    print("Dataset name:", base_self.__ds_name)
    print("Original public at paper:", base_self.__ds_paper_link)
    print("Dataset link:", base_self.__ds_link)
    print("Total size on disk (MB):", base_self.__ds_size)
    print("Total ",len(base_self.__ds_fts), "features: ")
    print(base_self.__ds_fts)
    print("Total ",len(base_self.__ds_label), "classes: ")
    print(base_self.__ds_label)
    print("==================================================================================")

  def Show_basic_analysis(base_self):
    """
    Show basic analysis of the dataset.
    Parameters:
      None
    Returns:
      None
    """
    print("===================== Show basic analysis of data frame ==========================")
    print("============================= Dataframe be like ==================================")
    print('\n' + tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql'))
    print("================================== Data info =====================================")
    print(base_self.__data_df.info())
    print("============================== Label distribution ================================")
    print(base_self.__data_df[base_self.__target_variable].value_counts())
    print("==================================================================================")
    
  #=========================================================================================================================================
  
  def To_dataframe(base_self):
    """
    Return the dataset as a pandas DataFrame.
    Parameters:
      None
    Returns:
      A pandas DataFrame.
    """
    return base_self.__data_df

  #=========================================================================================================================================
  
  def Split_data(base_self, target_variable=None):
    """
    Return the dataset as features and label.
    Parameters:
      target_variable = string:
        Name of column contains the data label. Default is None.
    Returns:
      X = np.array:
        Features.
      y = np.array:
        Label.
    """
    if target_variable is None or target_variable not in base_self.__label_fts_names:
      raise ValueError(f"Invalid data label!!! Select one of list labels: {base_self.__label_fts_names}")
    print("============================== Begin Split File ==================================")
    df = base_self.__data_df
    base_self.__print("============================== Dataframe be like =================================")
    base_self.__print(f"\n {tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql')}")

    X = df.drop(columns = base_self.__label_fts_names, axis=1).to_numpy()
    y = df[target_variable].to_numpy()
      
    base_self.__print(f"Features shape:{X.shape}")
    base_self.__print(f"Label count: {np.unique(y)}")

    return X, y
  
  #=========================================================================================================================================

  def To_csv(base_self, path = 'Default'):
    """
    Save the dataset as a csv file.
    Parameters:
      path = str:
        Path to save the csv file. Default is current working directory.
    Returns:
      None
    """
    if path == 'Default':
      path = base_self.__data_dir
      data_file = os.path.join(path, ("data_" + base_self.__ds_name + ".csv"))
    else:
      data_file = path
    if os.path.exists(data_file) == True:
      print("File is already exists at path:", data_file)
    else:
      base_self.__data_df.to_csv(data_file, index=False)
      print("File saved at:", data_file)
    return