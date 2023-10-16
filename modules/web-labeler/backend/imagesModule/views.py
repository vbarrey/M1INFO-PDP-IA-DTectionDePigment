from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from django.core.serializers import serialize
from .models import Image
from .form import ImageUploadForm
from django.db.models import Q

def accueil(request):
    toLabel = Image.objects.all().filter(workingStatus="Unlabeled")
    inProgress = Image.objects.all().filter(workingStatus="InProgress")
    return render(request, 'accueil.html', {'toLabel' : toLabel, 'inProgress' : inProgress})

def upload(request):
    if request.method == 'POST':
        return uploadRequest(request)
    else :
        form = ImageUploadForm()
        return render(request, 'upload.html', {'form': form})

def labeler(request):
    if len(Image.objects.all().filter(~Q(workingStatus="Labeled"))) != 0:
        image = Image.objects.all().filter(~Q(workingStatus="Labeled"))[0]
        url="/labeler/"+str(image.id)
        return redirect(url)
    else:
        return render(request, "labeler2.html", {})
    

def labelerId(request, id):
    if request.method == 'POST':
        return uploadJson(request, id)
    else :
        image = Image.objects.get(id=id)
        array_result = serialize('json', [image], ensure_ascii=False)
        json = array_result[1:-1]
        return render(request, "labeler2.html", {'image': image, 'json':json})

def uploadRequest(request):                                      
    images = request.FILES.getlist('images')
    imgUploaded = []
    imgNotUploaded = []
    
    
    for file in images:
        img, created = Image.objects.get_or_create(title=file.name, type=file.content_type, defaults={'img':file, 'workingStatus':"Unlabeled"})
        if created:
            imgUploaded.append(img.title)
        else:
            imgNotUploaded.append(img.title)

        
    context = {
        'imgUploaded' : imgUploaded,
        'imgNotUploaded' : imgNotUploaded
    }
    return render(request, 'upload.html', context)

def uploadJson(request, id):
    print(request.POST)
    image = Image.objects.get(id=id)
    check = request.POST.getlist('workingStatus')
    if request.FILES:
        json = (request.FILES.getlist('jsonFile')[0])
    else :
        json = (request.POST.getlist('jsonFile')[0])
    if (len(check) > 0) and check[0] == "on":
        image.workingStatus = "Labeled"
    else :
        image.workingStatus = "InProgress"

    image.json = json
    image.save()
    return redirect('/labeler/')
