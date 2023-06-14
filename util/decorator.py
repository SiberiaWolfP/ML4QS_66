def check_empty_args(func):
    def wrapper(*args, **kwargs):
        if len(args) == 0:
            print("the length of one of paras is 0, func is quiting")
            return
        return func(*args, **kwargs)
    return wrapper
