from django.db import models
from django.contrib.auth.models import User
from PIL import Image



# Extending User Model Using a One-To-One Link
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    avatar = models.ImageField(default='default.jpg', upload_to='profile_images')
    bio = models.TextField()

    def __str__(self):
        return self.user.username

    # resizing images
    def save(self, *args, **kwargs):
        super().save()

        img = Image.open(self.avatar.path)

        if img.height > 100 or img.width > 100:
            new_img = (100, 100)
            img.thumbnail(new_img)
            img.save(self.avatar.path)

    
from django.db import models

class UserImageModel(models.Model):
    image = models.ImageField(upload_to='images/')
    label = models.CharField(max_length=20, default='data')

    def __str__(self):
        return self.label


from django.db import models

class Patient_info(models.Model):
    symptoms = models.IntegerField()
    Alzheimer_Disease = models.IntegerField()
    Overlapping_Symptom = models.IntegerField()
    label = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.symptoms} - {self.Alzheimer_Disease} - {self.Overlapping_Symptom} => {self.label}"



# models.py
from django.db import models

class AudioPrediction(models.Model):
    audio_file = models.FileField(upload_to='audio_files/')
    prediction = models.CharField(max_length=50)
    upload_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.prediction
