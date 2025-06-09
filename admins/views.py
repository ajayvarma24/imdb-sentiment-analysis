from django.shortcuts import render
from django.contrib import messages
from users.forms import UserRegistrationForm
from users.models import UserRegistrationModel

# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def AdminHome(request):
    return render(request, 'admins/AdminHome.html')

def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/viewregisterusers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/viewregisterusers.html',{'data':data})
    

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .models import InputData

TEXT_FILE_PATH = 'saved_inputs.txt'

def dynamic_input_view(request):
    saved_inputs = InputData.objects.all()
    return render(request, 'admins/dynamic_input_form.html', {'saved_inputs': saved_inputs})

@csrf_exempt
def save_inputs(request):
    if request.method == 'POST':
        input_list = json.loads(request.POST.get('inputs', '[]'))

        with open(TEXT_FILE_PATH, 'a') as file:
            for input_text in input_list:
                # Save to database
                obj = InputData.objects.create(text=input_text)
                # Save to text file
                file.write(f'{obj.id}: {input_text}\n')

        return JsonResponse({'message': 'Inputs saved successfully!'})
    return JsonResponse({'message': 'Invalid request method'}, status=400)

@csrf_exempt
def delete_input(request):
    if request.method == 'POST':
        input_id = request.POST.get('id')
        try:
            input_obj = InputData.objects.get(id=input_id)
            input_obj.delete()

            # Rewrite the text file after deletion
            all_inputs = InputData.objects.all()
            with open(TEXT_FILE_PATH, 'w') as file:
                for item in all_inputs:
                    file.write(f'{item.id}: {item.text}\n')

            return JsonResponse({'message': 'Input deleted successfully'})
        except InputData.DoesNotExist:
            return JsonResponse({'message': 'Input not found'}, status=404)
    return JsonResponse({'message': 'Invalid request method'}, status=400)
