{{ name | escape | underline}}

.. currentmodule:: {{ module }}
.. tip::
      |{{ name }}|


.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   {% if methods %}
   .. epigraph:: {{ _('**Methods:**') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {% if item != '__init__' %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. epigraph:: {{ _('**Attributes:**') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
