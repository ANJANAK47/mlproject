import sys
import logging

def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return "Error occurred in python script [{0}] line [{1}] message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # so we can see the log
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by Zero")
        # pass the sys module so error_message_detail can call sys.exc_info()
        raise CustomException(e, sys)
