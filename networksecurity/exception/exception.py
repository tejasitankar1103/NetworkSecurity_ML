import sys

from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename

    
    def __str__(self):
        return "Error occured in python scipt name [{0}] line number [{1}] error message with message [{2}]".format(
            self.file_name,self.lineno,str(self.error_message)
        )
    

# if __name__ == '__main__':
#     try:
#         logger.logging.info("enter try blocl")
#         a=1/0
        
#         print("This will not be printed",a)

#     except Exception as e:
#         logger.logging.error("ERORR")
#         raise NetworkSecurityException(e,sys)
    