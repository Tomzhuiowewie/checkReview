import pandas as pd
import configparser
import pymysql
from sqlalchemy import Column, String, Integer, create_engine, and_ # type: ignore
from sqlalchemy.orm import sessionmaker,declarative_base # type: ignore
from sqlalchemy import create_engine
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError, OperationalError

_engine = None
def get_engine(host,port,user, password, database):
    global _engine
    if _engine is None:
        _engine = create_engine(
            f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}',
            pool_pre_ping=True  # 防止连接失效
        )
    return _engine

# 创建 SQLAlchemy 基类
Base = declarative_base()

class ShDemandDeclaration(Base):
    """
    定义 SQLAlchemy 模型类，映射到 sh_demand_declaration 表。
    该表存储需求申报相关信息。
    """
    __tablename__ = 'sh_demand_declaration'
    # 主键 ID
    id = Column(Integer, primary_key=True)
    # 场景名称
    scene_name = Column(String(255))
    # 申报单位
    org_name = Column(String(255))
    # 联系人姓名
    contact_name = Column(String(255))
    # 需求级别
    level_type = Column(Integer)
    # 场景类型
    scene_type = Column(String(255))
    # 专业领域
    prfs_field_name = Column(String(255))
    # 需求简介/工作背景
    work_background = Column(String(1255))
    # 需求简介/需求描述
    requirement_description = Column(String(1255))
    # 需求简介/预期效果
    expected_effect = Column(String(1255))
    # 是否删除
    is_del = Column(Integer)

class ShResults(Base):
    """
    定义SQLAlchemy模型类，映射到sh_results表。
    该表存储结果相关信息。
    """
    __tablename__ = 'sh_results'
    # 主键 ID
    id = Column(Integer, primary_key=True)
    # 场景名称
    scene_name = Column(String(255))
    # 申报单位
    org_name = Column(String(255))
    # 联系人姓名
    contact_name = Column(String(255))
    # 参考类型
    ref_type = Column(String(255))
    # 参考标题
    ref_title = Column(String(255))
    # 专业领域名称
    prfs_field_name = Column(String(255))
    # 建设背景
    constr_background = Column(String(1255))
    # 建设必要性
    constr_necessity = Column(String(1255))
    # 建设目标
    constr_target = Column(String(1255))
    # 建设过程
    constr_process = Column(String(1255))
    # 成果成效
    achievement = Column(String(1255))
    # 场景概述
    scene_overview = Column(String(1255))
    # 是否删除
    is_del = Column(Integer)
    # 状态
    status = Column(Integer)
    # 唯一标识
    # p_up_id = Column(String(1255))
    # 版本信息
    uuid = Column(String(255))
    # 创建时间
    # create_time = Column(Integer)
    # 更新时间
    # update_time = Column(Integer)

class ReviewAgingAward(Base):
    """
    定义 SQLAlchemy 模型类，映射到 review_aging_award 表。
    该表存储结果相关信息。
    """
    __tablename__ ='review_aging_award'
    # 主键 ID
    id = Column(Integer, primary_key=True)
    # 成果 ID
    result_id = Column(Integer)
    # 奖项名称
    prize_name = Column(String(255))
    # 获奖时间
    task_id = Column(Integer)

class ReviewAwardSituation(Base):
    """
    定义 SQLAlchemy 模型类，映射到 review_award_situation 表。
    该表存储结果相关信息。
    """
    __tablename__ ='review_award_situation'
    # 主键 ID
    id = Column(Integer, primary_key=True)
    # 成果 ID
    result_id = Column(Integer)
    # 奖项名称
    prize_name = Column(String(255))
    # 获奖时间
    task_id = Column(Integer)

class ReviewTask(Base):
    """
    定义 SQLAlchemy 模型类，映射到 review_task 表。
    该表存储结果相关信息。
    """
    __tablename__ ='review_task'
    # 主键 ID
    id = Column(Integer, primary_key=True)
    # 奖项名称
    title = Column(String(255))

class ShResultsP(Base):
    """
    定义SQLAlchemy模型类，映射到sh_results表。
    该表存储结果相关信息。
    """
    __tablename__ = 'sh_results_p'
    # 主键 ID
    id = Column(Integer, primary_key=True)
    # 场景名称
    scene_name = Column(String(255))
    # 申报单位
    org_name = Column(String(255))
    # 联系人姓名
    contact_name = Column(String(255))
    # 参考类型
    ref_type = Column(String(255))
    # 参考标题
    ref_title = Column(String(255))
    # 专业领域名称
    prfs_field_name = Column(String(255))
    # 建设背景
    constr_background = Column(String(1255))
    # 建设必要性
    constr_necessity = Column(String(1255))
    # 建设目标
    constr_target = Column(String(1255))
    # 建设过程
    constr_process = Column(String(1255))
    # 成果成效
    achievement = Column(String(1255))
    # 是否删除
    is_del = Column(Integer)
    # 状态
    status = Column(Integer)
    # 版本信息
    uuid = Column(String(255))

def read_config(file_path, section='database'):
    """
    从配置文件中读取数据库配置信息。
    :param file_path: 配置文件路径
    :return: 包含数据库配置信息的字典，如果读取失败则返回空字典
    """
    config = configparser.ConfigParser()
    try:
        config.read(file_path, encoding='utf-8')
        return config[section]
    except (KeyError, FileNotFoundError) as e:
        print(f"读取配置文件出错: {e}")
        return {}

def execute_sql(
    host,
    port,
    user,
    password,
    database,
    model,
    ids: Optional[List[Any]] = None,
    filters: Optional[Dict[str, Optional[int]]] = None,
    included_columns: Optional[List[str]] = None
):
    """
    通用 SQL 查询函数，支持过滤与字段选择，性能优化版。
    """
    try:
        # 数据库连接
        try:
            engine = get_engine(host, port, user, password, database)
            Session = sessionmaker(bind=engine)
        except OperationalError as conn_err:
            print(f"数据库连接失败: {conn_err}")
            return pd.DataFrame()

        with Session() as session:
            try:
                # 字段选择
                if included_columns:
                    query = session.query(*[getattr(model, col) for col in included_columns])
                else:
                    query = session.query(model)

                # 构建查询条件
                conditions = []
                if filters:
                    for attr, value in filters.items():
                        if '__gte' in attr:
                            field = attr.replace('__gte', '')
                            conditions.append(getattr(model, field) >= value)
                        elif '__lte' in attr:
                            field = attr.replace('__lte', '')
                            conditions.append(getattr(model, field) <= value)
                        else:
                            conditions.append(getattr(model, attr) == value)
                if ids is not None:
                    conditions.append(model.id.in_(ids))
                if conditions:
                    query = query.filter(and_(*conditions))

                # 分批拉取
                query = query.yield_per(5000)

                results = query.all()

            except SQLAlchemyError as query_err:
                print(f"查询执行失败: {query_err}")
                return pd.DataFrame()

        # 转换为 DataFrame
        try:
            if included_columns:
                df = pd.DataFrame([{col: getattr(row, col, None) for col in included_columns} for row in results])
            else:
                df = pd.DataFrame([{col.name: getattr(row, col.name, None) for col in model.__table__.columns} for row in results])

            return df

        except Exception as df_err:
            print(f"结果处理失败: {df_err}")
            return pd.DataFrame()

    except Exception as e:
        print(f"执行 SQL 查询时出错: {e}")
        return pd.DataFrame()

def save_to_mysql(df_group, df_reason, dfs, data):
    db_config = read_config('./input/config.ini')
    if not db_config:
        return

    host = db_config.get('host')
    user = db_config.get('user')
    password = db_config.get('password')
    database = db_config.get('new_database')

    # 创建一个临时连接，用于检查数据库是否存在
    connection = pymysql.connect(host=host, user=user, password=password)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{database}'")
            result = cursor.fetchone()
            if not result:
                cursor.execute(f"CREATE DATABASE {database}")
                print(f"数据库 {database} 已创建。")
    except pymysql.MySQLError as e:
        print(f"数据库操作错误: {e}")
    finally:
        connection.close()

    # 创建数据库连接引擎
    engine = get_engine(host, user, password, database)

    try:
        # 保存 df_group 和 df_reason
        df_group.to_sql('repeat_group', engine, if_exists='replace', index=False, chunksize=1000)
        df_reason.to_sql('repeat_reason', engine, if_exists='replace', index=False, chunksize=1000)

        # 保存 dfs 字典中的其他 DataFrame
        for keyword, df in dfs.items():
            pd.DataFrame(df).to_sql(f"repeat_{keyword}", engine, if_exists='replace', index=False, chunksize=1000)

        # 保存字段权重、阈值信息，存为单独一张表，将所有内容存储到一张新表 data_config
        data_config = {
            'weights': data.get("weights", {}),
            'thresholds': data.get("thresholds", {}),
            'whole_threshold': data.get("whole_threshold", 80)
        }

        # 将 data_config 结构转为 DataFrame
        df_data_config = pd.DataFrame([{
            "weights": str(data_config['weights']),
            "thresholds": str(data_config['thresholds']),
            "whole_threshold": data_config['whole_threshold']
        }])

        # 创建新表并保存 data 配置
        df_data_config.to_sql('data_config', engine, if_exists='replace', index=False, chunksize=1000)

    except Exception as e:
        print(f"保存数据到数据库时出错: {e}")

if __name__ == "__main__":
    db_config_path = './input/config.ini'
    db_config = read_config(db_config_path)
    print("进入函数")
    df = execute_sql(db_config["host"],
                     db_config["port"],
                     db_config["user"],
                     db_config["password"],
                     db_config["database"],
                     ShResults,
                     filters={"is_del": None,"status": 2}
                     )