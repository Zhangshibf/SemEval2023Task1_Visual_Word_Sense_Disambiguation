import torch
import torch.nn as nn
from transformers import ViltModel
from transformers.models.vilt.modeling_vilt import ViltForImagesAndTextClassificationOutput
from transformers.modeling_utils import PreTrainedModel

class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # print(config)
        self.num_labels = config.num_labels
        self.vilt = ViltModel(config)

        # Classifier head
        num_images = config.num_images
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * num_images, config.hidden_size * num_images),
            nn.LayerNorm(config.hidden_size * num_images),
            nn.GELU(),
            nn.Linear(config.hidden_size * num_images, config.num_labels),
        )


    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        pixel_values = None,
        pixel_mask = None,
        head_mask = None,
        inputs_embeds = None,
        image_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(input_ids)
        # print(pixel_values.size())
        if pixel_values is not None and pixel_values.ndim == 4:
            # add dummy num_images dimension
            pixel_values = pixel_values.unsqueeze(1)

        if image_embeds is not None and image_embeds.ndim == 3:
            # add dummy num_images dimension
            image_embeds = image_embeds.unsqueeze(1)

        num_images = pixel_values.shape[1] if pixel_values is not None else None
        # print(num_images)
        if num_images is None:
            num_images = image_embeds.shape[1] if image_embeds is not None else None
        if num_images != self.config.num_images:
            raise ValueError(
                "Make sure to match the number of images in the model with the number of images in the input."
            )
        pooler_outputs = []
        hidden_states = [] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for i in range(num_images):
          # print(i)
          # print(input_ids)
          # print(pixel_values[:, i, :, :, :])
          
          # forward every image through the model
          outputs = self.vilt(
              input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids,
              pixel_values=pixel_values[:, i, :, :, :] if pixel_values is not None else None,
              pixel_mask=pixel_mask[:, i, :, :] if pixel_mask is not None else None,
              head_mask=head_mask,
              inputs_embeds=inputs_embeds,
              image_embeds=image_embeds[:, i, :, :] if image_embeds is not None else None,
              image_token_type_idx=1,
              output_attentions=output_attentions,
              output_hidden_states=output_hidden_states,
              return_dict=return_dict,
          )
          # print("="*20)
          # print(outputs)
          pooler_output = outputs.pooler_output if return_dict else outputs[1]
          # print("="*20)
          # print(pooler_output)
          pooler_outputs.append(pooler_output)
          if output_hidden_states:
              hidden_states.append(outputs.hidden_states)
          if output_attentions:
              attentions.append(outputs.attentions)

        pooled_output = torch.cat(pooler_outputs, dim=-1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # print(labels)
            loss = loss_fct(logits.view(-1, self.num_labels), labels)

        if not return_dict:
            output = (logits, hidden_states, attentions)
            return ((loss,) + output) if loss is not None else output

        return ViltForImagesAndTextClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )