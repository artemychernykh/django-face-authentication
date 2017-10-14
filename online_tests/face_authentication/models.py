from django.db import models

class FAUser(models.Model):
    username = models.CharField(max_length=128)
    token = models.CharField(max_length=128, blank=True, null=True)
    photo= models.ImageField(upload_to='users')

    def __str__(self):
        return '{}'.format(self.username)
