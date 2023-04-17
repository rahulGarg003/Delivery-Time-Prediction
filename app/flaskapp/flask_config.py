import secrets
from sys import exit
from dotenv import load_dotenv
import os

class Config(object):

    #setup App Secret Key
    SECRET_KEY = secrets.token_hex(20)
    load_dotenv()

class ProductionConfig(Config):

    #set debug flag
    DEBUG = False

config_dict = {
    'Production' : ProductionConfig
}