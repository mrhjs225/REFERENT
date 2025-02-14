import os


def get_checkpoint_num(model_dir):
    folder_list = os.listdir(model_dir)
    maximum_num = 0
    for folder in folder_list:
        if folder.startswith('checkpoint-'):
            folder_num = int(folder.split('-')[1])
            if folder_num > maximum_num:
                maximum_num = folder_num
    return str(maximum_num)


def run_no_reference():
    model_id_list = ['0', '1-1', '1-2', '1-3', '1-4', '3-1', '3-2', '3-3', '3-4', '4-1']
    model_id_list = ['1-1']
    num_of_ref = '0'

    model_name = 't5-base'
    for model_id in model_id_list:
        print('====== ref {}, model {} ======'.format(num_of_ref, model_id))
        test_dataset_dir = '../data/codeforce/test/test_model_{}.json'.format(model_id)
        pretrained_model_dir = '../models/codechef/codechef_{}/'.format(model_id)
        pretrained_model_dir = '{}checkpoint-{}/'.format(pretrained_model_dir, get_checkpoint_num(pretrained_model_dir))
        test_result_dir = '../test_result/codeforce/0/model_{}'.format(model_id)
        os.system('rm -rf {}'.format(test_result_dir))
        os.system('python tfix_testing.py -mn {} -md {} -dd {} -lm {} -ta True -bid {}'.format(model_name, test_result_dir, test_dataset_dir, pretrained_model_dir, 0))


def run_reference(num_of_ref):
    model_id_list = ['0', '1-1', '1-2', '1-3', '1-4', '3-1', '3-2', '3-3', '3-4', '4-1']
    model_name = 't5-base'
    epoch = 30
    batch = 32
    use_pretrain_model = 'True'

    for model_id in model_id_list:
        print('====== ref {}, model {} ======'.format(num_of_ref, model_id))
        test_dataset_dir = '../data/codeforce/test/test_model_{}.json'.format(model_id)
        pretrained_model_dir = '../models/codechef/codechef_{}/'.format(model_id)
        pretrained_model_dir = '{}checkpoint-{}/'.format(pretrained_model_dir, get_checkpoint_num(pretrained_model_dir))
        train_dataset_dir = '../data/codeforce/finetuning/{}/finetuning_{}_model_{}.json'.format(num_of_ref, num_of_ref, model_id)
        save_model_dir = '../models/codeforce/ref_{}/model_{}/'.format(num_of_ref, model_id)

        os.system('rm -rf {}'.format(save_model_dir))
        os.system('python tfix_training.py -e {} -bs {} -mn {} -md {} -lm {} -dd {} -bid {} -pt {}'.format(epoch, batch, model_name, save_model_dir, pretrained_model_dir, train_dataset_dir, 0, use_pretrain_model))

        save_model_dir += 'checkpoint-{}/'.format(get_checkpoint_num(save_model_dir))
        test_result_dir = '../test_result/codeforce/ref_{}/model_{}'.format(num_of_ref, model_id)
        os.system('rm -rf {}'.format(test_result_dir))
        os.system('python tfix_testing.py -mn {} -md {} -dd {} -lm {} -ta True -bid {}'.format(model_name, test_result_dir, test_dataset_dir, save_model_dir, 0))

def run_rq3_ref(num_of_ref):
    model_id_list = ['abla_1', 'abla_2', 'abla_4']
    model_name = 't5-base'
    epoch = 30
    batch = 32
    use_pretrain_model = 'True'

    for model_id in model_id_list:
        print('====== ref {}, model {} ======'.format(num_of_ref, model_id))
        if model_id == 'abla_1':
            test_dataset_dir = '../data/codeforce/test/test_model_{}.json'.format('4-1')
            pretrained_model_dir = '../models/t5base/'
        elif model_id == 'abla_2':
            train_dataset_dir = '../data/codeforce/finetuning/{}/finetuning_{}_model_{}.json'.format(num_of_ref, num_of_ref, '0')
            save_model_dir = '../models/codeforce/abla_2/'
            os.system('rm -rf {}'.format(save_model_dir))
            os.system('python tfix_training.py -e {} -bs {} -mn {} -md {} -lm {} -dd {} -bid {} -pt {}'.format(epoch, batch, model_name, save_model_dir, pretrained_model_dir, train_dataset_dir, 0, use_pretrain_model))

            test_dataset_dir = '../data/codeforce/test/test_model_{}.json'.format('4-1')
            pretrained_model_dir += 'checkpoint-{}/'.format(get_checkpoint_num(save_model_dir))
        elif model_id == 'abla_4':
            train_dataset_dir = '../data/codeforce/finetuning/{}/finetuning_{}_model_{}.json'.format(num_of_ref, num_of_ref, '4-1')
            save_model_dir = '../models/codeforce/abla_4/'
            os.system('rm -rf {}'.format(save_model_dir))
            os.system('python tfix_training.py -e {} -bs {} -mn {} -md {} -lm {} -dd {} -bid {} -pt {}'.format(epoch, batch, model_name, save_model_dir, pretrained_model_dir, train_dataset_dir, 0, use_pretrain_model))
            test_dataset_dir = '../data/codeforce/test/test_model_{}.json'.format('4-1')
            pretrained_model_dir += 'checkpoint-{}/'.format(get_checkpoint_num(save_model_dir))

        train_dataset_dir = '../data/codeforce/finetuning/{}/finetuning_{}_model_{}.json'.format(num_of_ref, num_of_ref, '4-1')
        save_model_dir = '../models/codeforce/ref_{}/model_{}/'.format(num_of_ref, model_id)

        os.system('rm -rf {}'.format(save_model_dir))
        os.system('python tfix_training.py -e {} -bs {} -mn {} -md {} -lm {} -dd {} -bid {} -pt {}'.format(epoch, batch, model_name, save_model_dir, pretrained_model_dir, train_dataset_dir, 0, use_pretrain_model))

        save_model_dir += 'checkpoint-{}/'.format(get_checkpoint_num(save_model_dir))
        test_result_dir = '../test_result/codeforce_ref_{}_model_{}'.format(num_of_ref, model_id)
        os.system('rm -rf {}'.format(test_result_dir))
        os.system('python tfix_testing.py -mn {} -md {} -dd {} -lm {} -ta True -bid {}'.format(model_name, test_result_dir, test_dataset_dir, save_model_dir, 0))

if __name__ == '__main__':
    # RQ 1,2
    run_no_reference()
    # run_reference('10')
    # run_reference('30')
    # run_reference('50')

    #  RQ3
    # run_rq3_ref('50')