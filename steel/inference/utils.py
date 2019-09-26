#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def get_encoded_pixels():
    # make preidctions for segmentation model

    predictions = []
    for i, batch in enumerate(tqdm(testset)):
        fnames, images = batch
        #print('images', images.shape)
        images = images.cuda()
        batch_preds = 0
        probabilities = []
        for model in models:
            model = model.cuda()
            for k, (a, inv_a) in enumerate(augment):
                    logit = model(a(images))
                    p = inv_a(torch.sigmoid(logit))

                    if k ==0:
                        probability  = p**0.5
                    else:
                        probability += p**0.5
            probability = probability/len(augment)
            probabilities.append(probability)

            batch_preds+=probability

        batch_preds = batch_preds.data.cpu().numpy()
        #print(batch_preds.shape)
        for fname, preds in zip(fnames, batch_preds):
            for cls, pred in enumerate(preds):
                #print(cls)
                pred, num = post_process(pred, threshold_pixel[cls], min_size[cls])
                rle = mask2rle(pred)
                name = fname + f"_{cls+1}"
                predictions.append([name, rle])


    df_segmentation = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
