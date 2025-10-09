import io
import torch
import numpy as np
import boto3
from PIL import Image, ImageSequence, ImageOps

def awss3_save_file(client, bucket, key, buff):
    client.put_object(
            Body = buff,
            Key = key, 
            Bucket = bucket)

def awss3_load_file(client, bucket, key):
    outfile = io.BytesIO()
    client.download_fileobj(bucket, key, outfile)
    outfile.seek(0)
    return outfile

def awss3_init_client(region="us-east-1", ak=None, sk=None, session=None, endpoint_url=None):
    client = None
    client_kwargs = {'region_name': region}
    
    # Добавляем endpoint если указан
    if endpoint_url:
        client_kwargs['endpoint_url'] = endpoint_url
    
    if (ak == None and sk == None) and session == None:
        client = boto3.client('s3', **client_kwargs)
    elif (ak != None and sk != None) and session == None:
        client_kwargs.update({
            'aws_access_key_id': ak, 
            'aws_secret_access_key': sk
        })
        client = boto3.client('s3', **client_kwargs)
    elif (ak != None and sk != None) and session != None:
        client_kwargs.update({
            'aws_access_key_id': ak, 
            'aws_secret_access_key': sk, 
            'aws_session_token': session
        })
        client = boto3.client('s3', **client_kwargs)
    else:
        client = boto3.client('s3', **client_kwargs)
    return client


# SaveImageToS3R2
class SaveImageToS3R2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "images": ("IMAGE",), 
                             "region": ("STRING", {"multiline": False, "default": "us-east-1"}),
                             "endpoint_url": ("STRING", {"multiline": False, "default": ""}),
                             "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                             "aws_sk": ("STRING", {"multiline": False, "default": ""}),
                             "session_token": ("STRING", {"multiline": False, "default": ""}),
                             "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                             "pathname": ("STRING", {"multiline": False, "default": "pathname for file"}),
                             "format": (["PNG", "JPG"], {"default": "JPG"}),
                             "jpg_quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1})
                             },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ()
    FUNCTION = "save_image_to_s3r2"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def save_image_to_s3r2(self, images, region, endpoint_url, aws_ak, aws_sk, session_token, s3_bucket, pathname, format, jpg_quality, prompt=None, extra_pnginfo=None):
        try:
            client = awss3_init_client(region, aws_ak, aws_sk, session_token, endpoint_url)
            results = list()
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                # Конвертируем в RGB для JPG (JPG не поддерживает альфа-канал)
                if format == "JPG" and img.mode in ('RGBA', 'LA', 'P'):
                    # Создаем белый фон для прозрачных областей
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif format == "JPG" and img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_byte_arr = io.BytesIO()
                
                if format == "JPG":
                    # Оптимальные настройки для JPG сжатия
                    img.save(img_byte_arr, format='JPEG', quality=jpg_quality, optimize=True, progressive=True)
                    file_extension = "jpg"
                else:
                    img.save(img_byte_arr, format='PNG', optimize=True)
                    file_extension = "png"
                
                img_byte_arr.seek(0)  # Сбрасываем указатель в начало буфера
                
                # Получаем размер данных для отладки
                data = img_byte_arr.getvalue()
                print(f"Размер данных изображения ({format}): {len(data)} байт")
                
                filename = "%s_%i.%s" % (pathname, batch_number, file_extension)
                awss3_save_file(client, s3_bucket, filename, data)
                results.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "output"
                })
            return { "ui": { "images": results } }
        except Exception as e:
            print(f"Ошибка при сохранении в S3/R2: {e}")
            return { "ui": { "images": [] } }

# LoadImageFromS3R2
class LoadImageFromS3R2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"region": ("STRING", {"multiline": False, "default": "us-east-1"}),
                             "endpoint_url": ("STRING", {"multiline": False, "default": ""}),
                             "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                             "aws_sk": ("STRING", {"multiline": False, "default": ""}),
                             "session_token": ("STRING", {"multiline": False, "default": ""}),
                             "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                             "pathname": ("STRING", {"multiline": False, "default": "pathname for file"})
                             } 
                }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (False, )
    FUNCTION = "load_image_from_s3r2"
    CATEGORY = "image"

    def load_image_from_s3r2(self, region, endpoint_url, aws_ak, aws_sk, session_token, s3_bucket, pathname):
        client = awss3_init_client(region, aws_ak, aws_sk, session_token, endpoint_url)
        img = Image.open(awss3_load_file(client, s3_bucket, pathname))
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                # Для JPG файлов создаем пустую маску (JPG не поддерживает альфа-канал)
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

# if __name__ == '__main__':
#     client = awss3_init_client()
#     awss3_save_file(client, "test-bucket", "test.jpg", awss3_load_file(client, "test-bucket", "test"))
