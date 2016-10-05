with open('temp.csv', 'w') as f1:
    with open('./tianchi_fresh_comp_train_user.csv') as f:
        context = f.readlines()
        i = 0
        for line in context:
            if i > 5000:
                break
            i = i + 1
            f1.write(line)
