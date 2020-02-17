
import tensorflow as tf 
from src.utils.config import process_config
from src.utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    
    args = get_args()
    print(args)

#    config = process_config(args.config)
#    print(config)


if __name__ == "__main__":
    main()