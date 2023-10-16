from django.contrib import admin
from .models import Image

# Register your models here.
class imageAdmin(admin.ModelAdmin):
    list_display = ["id", "title", "type", "img", "workingStatus", "json"]

admin.site.register(Image, imageAdmin)
