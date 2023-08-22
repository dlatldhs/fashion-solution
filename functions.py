import lib_import

def get_shoulder_len( shoulder, hip ):
    
    sh_diff = shoulder - hip
    
    if sh_diff < 0 :
        return -1
    
    else :
        if sh_diff > 0 and sh_diff < 41:
            result = 1
            print('bad')
        elif sh_diff > 40 and sh_diff < 61:
            result = 2
            print('normal')
        elif sh_diff > 60 and sh_diff < 81:
            result = 3
            print('good')
        else:
            result = 4
            print('very good')
    
    print(sh_diff)

    return sh_diff,result