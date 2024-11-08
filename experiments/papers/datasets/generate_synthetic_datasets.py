
# Generate the synthetic datasets used in experiments

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from core.bn import BN


def values_dataset_sachs_c():  # 10m rows
    bn = BN.read(EXPTS_DIR + '/bn/xdsl/sachs_c.xdsl')
    data = bn.generate_cases(10000000).astype('float32')
    data = Pandas(df=data)
    data.write(EXPTS_DIR + '/datasets/sachs_c.data.gz', sf=6,
               preserve=False)
    print(data.dstype)
    print(data.df.tail())


def values_dataset_covid_c():  # 10M rows
    bn = BN.read(EXPTS_DIR + '/bn/xdsl/covid_c.xdsl')
    data = bn.generate_cases(10000000).astype('float32')
    data = Pandas(df=data)
    data.write(EXPTS_DIR + '/datasets/covid_c.data.gz', sf=6,
               preserve=False)
    print(data.dstype)
    print(data.df.tail())


def values_dataset_building_c():  # 1M rows
    bn = BN.read(EXPTS_DIR + '/bn/xdsl/building_c.xdsl')
    data = bn.generate_cases(1000000).astype('float32')
    data = Pandas(df=data)
    data.write(EXPTS_DIR + '/datasets/building_c.data.gz', sf=6,
               preserve=False)
    print(data.dstype)
    print(data.df.tail())


def values_dataset_ecoli70_c():  # 1M rows
    bn = BN.read(EXPTS_DIR + '/bn/xdsl/ecoli70_c.xdsl')
    data = bn.generate_cases(1000000).astype('float32')
    data = Pandas(df=data)
    data.write(EXPTS_DIR + '/datasets/ecoli70_c.data.gz', sf=6,
               preserve=False)
    print(data.dstype)
    print(data.df.tail())


def values_dataset_niab_c():  # 1M rows
    bn = BN.read(EXPTS_DIR + '/bn/xdsl/magic-niab_c.xdsl')
    data = bn.generate_cases(1000000).astype('float32')
    data = Pandas(df=data)
    data.write(EXPTS_DIR + '/datasets/magic-niab.data_c.gz', sf=6,
               preserve=False)
    print(data.dstype)
    print(data.df.tail())


def values_dataset_irri_c():  # 1M rows
    bn = BN.read(EXPTS_DIR + '/bn/xdsl/magic-irri_c.xdsl')
    data = bn.generate_cases(1000000).astype('float32')
    data = Pandas(df=data)
    data.write(EXPTS_DIR + '/datasets/magic-irri_c.data.gz', sf=6,
               preserve=False)
    print(data.dstype)
    print(data.df.tail())


def values_dataset_arth150_c():  # 1M rows
    bn = BN.read(EXPTS_DIR + '/bn/xdsl/arth150_c.xdsl')
    data = bn.generate_cases(1000000).astype('float32')
    data = Pandas(df=data)
    data.write(EXPTS_DIR + '/datasets/arth150_c.data.gz', sf=6,
               preserve=False)
    print(data.dstype)
    print(data.df.tail())
