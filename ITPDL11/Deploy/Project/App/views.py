from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from .forms import UserImageForm
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.views import View
from django.contrib.auth.decorators import login_required 
from django.contrib.auth import logout as auth_logout
import numpy as np
import joblib
from .forms import RegisterForm, LoginForm, UpdateUserForm, UpdateProfileForm
from django.contrib.auth import authenticate,login,logout
from .models import UserImageModel
import numpy as np
from tensorflow import keras
from PIL import Image,ImageOps

import pyttsx3
import time


def home(request):
    return render(request, 'users/home.html')

@login_required(login_url='users-register')


def index(request):
    return render(request, 'app/index.html')

class RegisterView(View):
    form_class = RegisterForm
    initial = {'key': 'value'}
    template_name = 'users/register.html'

    def dispatch(self, request, *args, **kwargs):
        # will redirect to the home page if a user tries to access the register page while logged in
        if request.user.is_authenticated:
            return redirect(to='/')

        # else process dispatch as it otherwise normally would
        return super(RegisterView, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)

        if form.is_valid():
            form.save()

            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}')

            return redirect(to='login')

        return render(request, self.template_name, {'form': form})


# Class based view that extends from the built in login view to add a remember me functionality

class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if not remember_me:
            # set session expiry to 0 seconds. So it will automatically close the session after the browser is closed.
            self.request.session.set_expiry(0)

            # Set session as modified to force data updates/cookie to be saved.
            self.request.session.modified = True

        # else browser session will be as long as the session cookie time "SESSION_COOKIE_AGE" defined in settings.py
        return super(CustomLoginView, self).form_valid(form)


class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name = 'users/password_reset.html'
    email_template_name = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message = "We've emailed you instructions for setting your password, " \
                      "if an account exists with the email you entered. You should receive them shortly." \
                      " If you don't receive an email, " \
                      "please make sure you've entered the address you registered with, and check your spam folder."
    success_url = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')

from .models import Profile

def profile(request):
    user = request.user
    # Ensure the user has a profile
    if not hasattr(user, 'profile'):
        Profile.objects.create(user=user)
    
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        profile_form = UpdateProfileForm(request.POST, request.FILES, instance=request.user.profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile is updated successfully')
            return redirect(to='users-profile')
    else:
        user_form = UpdateUserForm(instance=request.user)
        profile_form = UpdateProfileForm(instance=request.user.profile)

    return render(request, 'users/profile.html', {'user_form': user_form, 'profile_form': profile_form})

from . models import UserImageModel,Patient_info
from . import forms
from .forms import UserImageForm
from PIL import Image
import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .preprocessing.noise_filters import apply_selected_filter
from .preprocessing.image_pipeline import load_image_for_preview, prepare_for_model, image_to_base64


# Human-readable labels for each filter key (used in preview.html)
_FILTER_DISPLAY_NAMES = {
    "skip":        "Skip (No noise removal)",
    "gaussian":    "Gaussian Filter",
    "median":      "Median Filter",
    "bilateral":   "Bilateral Filter",
    "nlm":         "Non-Local Means Denoising",
    "anisotropic": "Anisotropic Diffusion",
    "wavelet":     "Wavelet Denoising",
}


def _run_prediction(image_array):
    """
    Run the Keras LeNet model on a preprocessed image array and return
    the predicted class name and description string.

    Args:
        image_array: uint8 RGB numpy array (any size — will be resized internally).

    Returns:
        (predicted_class_str, description_str)
    """
    model = keras.models.load_model('App/keras_model.h5')
    data = prepare_for_model(image_array)
    classes = ["MildDemented", "ModerateDemented", "Non_Parkinson",
               "NonDemented", "Parkinson", "VeryMildDemented"]
    prediction = model.predict(data)
    idd = np.argmax(prediction)
    a = classes[idd]

    descriptions = {
        "MildDemented":     "This image Detected in Mild_Demented",
        "ModerateDemented": "This image Detected in Moderate_Demented",
        "Non_Parkinson":    "This image Detected in Non_Parkinson",
        "NonDemented":      "This image Detected in Non_Demented",
        "Parkinson":        "This image Detected in Parkinson",
        "VeryMildDemented": "This image Detected in Very_Mild_Impairment",
    }
    b = descriptions.get(a, "WRONG INPUT")
    return a, b


def Deploy_8(request):
    """
    Step 1 — Image upload only.

    GET  : Render the upload form (model.html).
    POST : Save the uploaded image, then redirect to the filter
           selection page (Step 2) passing the saved image ID.
    """
    if request.method == "POST":
        form = forms.UserImageForm(files=request.FILES)
        if form.is_valid():
            form.save()

        result1 = UserImageModel.objects.latest('id')
        return redirect('select_filter', image_id=result1.id)

    form = forms.UserImageForm()
    return render(request, 'app/model.html', {'form': form})


def select_filter(request, image_id):
    """
    Step 2 — Noise filter selection.

    GET  : Show the filter selection page with a thumbnail of the uploaded image.
    POST : Read chosen filter_type, then:
           - "skip"  → run prediction directly → output.html
           - other   → apply filter → preview.html (side-by-side comparison)
    """
    try:
        record = UserImageModel.objects.get(id=image_id)
    except UserImageModel.DoesNotExist:
        return redirect('Deploy_8')

    if request.method == "POST":
        filter_type = request.POST.get("filter_type", "skip").strip().lower()
        original_array = load_image_for_preview(record.image.path)

        if filter_type == "skip":
            # ── Direct prediction ─────────────────────────────────────────
            a, b = _run_prediction(original_array)
            record.label = a
            record.save()
            return render(request, 'App/output.html', {
                'obj': record, 'predict': a, 'predict1': b
            })

        else:
            # ── Apply filter → show preview before predicting ─────────────
            filtered_array = apply_selected_filter(original_array, filter_type)
            original_b64   = image_to_base64(original_array)
            filtered_b64   = image_to_base64(filtered_array)

            return render(request, 'app/preview.html', {
                'original_image':         original_b64,
                'filtered_image':         filtered_b64,
                'image_id':               record.id,
                'current_filter':         filter_type,
                'current_filter_display': _FILTER_DISPLAY_NAMES.get(filter_type, filter_type),
            })

    # GET — show filter selection page
    return render(request, 'app/filter_select.html', {
        'image_id':  record.id,
        'image_url': record.image.url,
    })


def apply_filter_ajax(request):
    """
    AJAX endpoint: re-apply a noise filter to an already-uploaded image.

    Called from the preview page when the user clicks "Apply Filter" after
    selecting a different filter from the dropdown — no page reload needed.

    POST body (JSON): { "image_id": <int>, "filter_type": <str> }

    Returns JSON: { "filtered_image": "<data-URI>" }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body)
        image_id    = int(body.get("image_id", 0))
        filter_type = str(body.get("filter_type", "skip")).strip().lower()
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    try:
        record = UserImageModel.objects.get(id=image_id)
    except UserImageModel.DoesNotExist:
        return JsonResponse({"error": "Image not found"}, status=404)

    original_array = load_image_for_preview(record.image.path)
    filtered_array = apply_selected_filter(original_array, filter_type)
    filtered_b64   = image_to_base64(filtered_array)

    return JsonResponse({"filtered_image": filtered_b64})


def predict_denoised(request):
    """
    Final prediction view — called when user clicks "Proceed to Prediction"
    on the preview page.

    POST form fields:
        image_id    : int — primary key of the saved UserImageModel record
        filter_type : str — chosen noise removal filter (or "skip")

    Applies the filter to the original image and feeds it to the Keras model.
    """
    if request.method != "POST":
        form = forms.UserImageForm()
        return render(request, 'app/model.html', {'form': form})

    try:
        image_id    = int(request.POST.get("image_id", 0))
        filter_type = str(request.POST.get("filter_type", "skip")).strip().lower()
    except (ValueError, TypeError):
        form = forms.UserImageForm()
        return render(request, 'app/model.html', {'form': form})

    try:
        record = UserImageModel.objects.get(id=image_id)
    except UserImageModel.DoesNotExist:
        form = forms.UserImageForm()
        return render(request, 'app/model.html', {'form': form})

    # Load, optionally denoise, then predict
    original_array  = load_image_for_preview(record.image.path)
    processed_array = apply_selected_filter(original_array, filter_type)

    a, b = _run_prediction(processed_array)

    record.label = a
    record.save()

    # Pass the filtered image as base64 only when a filter was actually applied
    filtered_b64 = image_to_base64(processed_array) if filter_type != "skip" else None

    return render(request, 'App/output.html', {
        'obj':            record,
        'predict':        a,
        'predict1':       b,
        'filtered_image': filtered_b64,
        'filter_name':    _FILTER_DISPLAY_NAMES.get(filter_type, filter_type) if filter_type != "skip" else None,
    })




def Database(request):
    models = UserImageModel.objects.all()
    return render(request, 'app/Database.html', {'models': models})

def mlDatabase(request):
    records = Patient_info.objects.all().order_by('-id')
    return render(request, 'app/ml database.html', {'records': records})

def logout_view(request):  
    auth_logout(request)
    return redirect('/')





Model = joblib.load('App/model_1.pkl')

    
from .models import Patient_info
import numpy as np


def model(request):
    if request.method == "POST":

        symptoms = int(request.POST.get("symptoms"))
        Alzheimer_Disease = int(request.POST.get("Alzheimer_Disease"))
        Overlapping_Symptom = int(request.POST.get("Overlapping_Symptom"))

        # ML Prediction
        final_features = np.array([[symptoms, Alzheimer_Disease, Overlapping_Symptom]])

        prediction = Model.predict(final_features)
        output = int(prediction[0])

        label = "normal" if output == 0 else "Parkinson"

        # SAVE to database
        record = Patient_info.objects.create(
            symptoms=symptoms,
            Alzheimer_Disease=Alzheimer_Disease,
            Overlapping_Symptom=Overlapping_Symptom,
            label=label
        )

        print("✔ Saved to DB:", record)

        return render(request, 'App/predict_out.html', {
            "prediction_text": label,
            "predict": output
        })

    return render(request, 'App/predict.html')


from django.shortcuts import render
from django.http import JsonResponse
# import random
# import json
import numpy as np
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
#from .models import Response, models

# Remove the comments to download additional nltk packages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST



               


