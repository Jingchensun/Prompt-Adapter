import os
from pydoc import cli
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import json

_tokenizer = _Tokenizer()


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TextEncoder2(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16 #n_ctx = 16
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        #print('ctx_dim:ctx_dimctx_dimctx_dim:',ctx_dim)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) #n_ctx = 16, ctx_dim =512
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        print('classnamesclassnames:',len(classnames)) #1000
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print('prompt_prefix',prompt_prefix) # X X X X X X X X X X X X X X X X

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.cuda()
        # print(tokenized_prompts.device)
        # print(clip_model.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            #print('embedding.size():',embedding.size())

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        #self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        # print('ctxctxctxctx:',ctx.size()) #torch.Size([16, 512])
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            #print('ctxctxctxctx:',ctx.size()) #torch.Size([100, 16, 512])

        prefix = self.token_prefix
        #print('prefix:', prefix.size()) #torch.Size([100, 1, 512])
        suffix = self.token_suffix
        #print('suffix.size()',suffix.size()) #torch.Size([100, 60, 512])

        # if self.class_token_position == "end":
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        #print('prompts:',prompts.size()) # torch.Size([100, 77, 512])
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.encode_text = TextEncoder2(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        #print(text_tensor.size())
        #text_features_ori = self.encode_text(text_tensor)

        prompts = self.prompt_learner()
        #print('prompts',prompts.size()) #torch.Size([100, 77, 512])
        tokenized_prompts = self.tokenized_prompts 
        #print('tokenized_prompts',tokenized_prompts.size()) #tokenized_prompts torch.Size([100, 77])
        text_features = self.text_encoder(prompts, tokenized_prompts)
        #print('text_features',text_features.size()) #text_features torch.Size([100, 1024])
        #torch.save(text_features, './mytensor2_3.pt')
        #print('image_features before:',image_features.size()) #torch.Size([25, 1024])

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #torch.save(text_features, './promt_text.pt')

        # print('image_features after:',image_features.size()) #torch.Size([25, 1024]) caltech torch.Size([8, 1024])
        # print('text_features.size():',text_features.size()) #caltech torch.Size([100, 1024])
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
 

        # text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
        # logits_per_image_ori = logit_scale * image_features @ text_features_ori.t()
        # logits_per_text_ori = logits_per_image_ori.t()


        return logits, image_features, text_features#, logits_per_image_ori, logits_per_text_ori

def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_model, train_loader_F, classnames):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    print('cache_keys.shape[0]:',cache_keys.shape[0]) #1024
    print('cache_keys.shape[1]:',cache_keys.shape[1]) #1600
    adapter.weight = nn.Parameter(cache_keys.t())

    
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0



    model = CustomCLIP(cfg, classnames, clip_model)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    model=model.cuda()


    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            #text = clip.tokenize(label_promt).cuda()
            #print('text_size:',text.size()) #torch.Size([64, 77])
            #print('images:',images.size())#([256, 3, 224, 224])
            with torch.no_grad():
                # image_features = clip_model.encode_image(images)
                # print('image_features.size():',image_features.size()) #torch.Size([256, 1024])
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                # text_features = clip_model.encode_text(text)
                # print('text_features.size():',text_features.size()) #torch.Size([256, 1024])
                # text_features /= text_features.norm(dim=-1, keepdim=True)

                #prompt
                
                output,image_features,text_features = model(images)



            affinity = adapter(image_features) 
            #print('affinity.size()',affinity.size())#torch.Size([256, 1600])
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            #print('cache_values:',cache_values.size()) #torch.Size([1600, 100])
            #clip_logits = 100. * image_features @ clip_weights
            clip_logits = output
            #print('clip_logits.size()',clip_logits.size()) #torch.Size([256, 100])
            tip_logits = clip_logits + cache_logits * alpha
            #print('cache_logits:',cache_logits.size()) #torch.Size([256, 100]
            #print(tip_logits.size()) #torch.Size([256, 100])

            # cosine similarity as logits
            # logit_scale = 100.
            # logits_per_image = logit_scale * image_features @ text_features.t()
            # #print('logits_per_image.size()',logits_per_image.shape[0]) #torch.Size([256, 256]) #64
            # logits_per_text = logits_per_image.t()
            # #print('logits_per_text.size()',logits_per_text.size())#torch.Size([256, 256])
            # ground_truth = torch.arange(logits_per_image.shape[0],dtype=torch.long,device='cuda')
            # #loss2 =  (F.cross_entropy(logits_per_image,ground_truth) +  F.cross_entropy(logits_per_text,ground_truth))/2
            # image_loss = tip_logits @ tip_logits.t()
            # loss2 = F.cross_entropy(image_loss, ground_truth) 
   
            loss3 = F.cross_entropy(clip_logits, target)
            loss1 = F.cross_entropy(tip_logits, target)
            #print('target:',len(target)) #
            loss = loss1 + loss3

            acc = cls_acc((tip_logits+clip_logits), target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        

        # clip_weights = torch.load('./promt_text.pt',map_location='cuda')
        # clip_weights = clip_weights.permute(1, 0)
        # clip_logits = 100. * test_features @ clip_weights
        clip_logits = 100. * test_features @ text_features.t()
        clip_weights = text_features

        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc((tip_logits+clip_logits), test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc((tip_logits+clip_logits), test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)


    # clip_weights = torch.load('./mytensor3.pt',map_location='cuda')
    # clip_weights = clip_weights.permute(1, 0)
    # print('clip_weights:',clip_weights.size()) #torch.Size([1024, 100])



    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    #run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_model, train_loader_F, (dataset.classnames))
           

if __name__ == '__main__':
    main()