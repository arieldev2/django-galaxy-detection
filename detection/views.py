from django.shortcuts import render
from .models import Detection
from django.views.generic import ListView, CreateView, DetailView, UpdateView, DeleteView


class DetectionListView(ListView):
	model = Detection
	template_name = 'detection/list.html'
	context_object_name = 'items' 


class DetectionDetailView(DetailView):
	model = Detection
	template_name = 'detection/detail.html'


class DetectionCreateView(CreateView):
	model = Detection
	template_name = 'detection/create.html'
	fields = ['title', 'img_original']


class DetectionUpdateView(UpdateView):
	model = Detection
	template_name = 'detection/create.html'
	fields = ['title', 'img_original']


class DetectionDeleteView(DeleteView):
	model = Detection
	template_name = 'detection/delete.html'
	success_url = '/'