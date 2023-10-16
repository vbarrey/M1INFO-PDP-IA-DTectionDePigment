from django.db import models
from django.core.files.storage import default_storage
from django.dispatch import receiver
from django.db.models.signals import post_delete
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os

class OverwriteStorage(FileSystemStorage):
    def get_available_name(self, name,  max_length=None):
        if self.exists(name):
            os.remove(os.path.join(settings.MEDIA_ROOT, name))
        return super(OverwriteStorage, self).get_available_name(name, max_length)

class Image(models.Model):
    title = models.CharField(max_length=20)
    type = models.CharField(max_length=20, default='png')
    img = models.ImageField(upload_to='images')
    workingStatus = models.CharField(default='Unlabeled', max_length=30)
    json = models.FileField(upload_to='jsons',  storage=OverwriteStorage(),  blank=True)

    class Meta:
        unique_together = ('title', 'type')

    @receiver(post_delete, sender=None)
    def delete_associated_files(sender, instance, **kwargs):
        """Remove all files of an image after deletion."""
        path = instance.img.name
        if path:
            default_storage.delete(path)


