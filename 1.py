

def a(c):
    str = ''
    for i, char in enumerate(c):
        if char == 'e':
            if i + 1 < len(c) and c[i + 1] != 'e':
                str = str + c[i]
            if i == len(c) -1:
                str = str + c[i]
        else:
            str = str + c[i]
    print(str)


a('abceeeessseeeebbdrreddsssaaaeeeee')
