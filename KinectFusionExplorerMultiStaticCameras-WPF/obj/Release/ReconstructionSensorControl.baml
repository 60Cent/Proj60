   M S B A M L   `   `   `  ���� O  Kmscorlib, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e0891&clr-namespace:System;assembly=mscorlibSystem  S OMicrosoft.Kinect.Toolkit, Version=1.8.0.0, Culture=neutral, PublicKeyToken=nulleHclr-namespace:Microsoft.Kinect.Toolkit;assembly=Microsoft.Kinect.ToolkitMicrosoft.Kinect.Toolkit e aKinectFusionExplorerMultiStaticCameras-WPF, Version=0.0.0.0, Culture=neutral, PublicKeyToken=nullm;clr-namespace:Microsoft.Samples.Kinect.KinectFusionExplorer-Microsoft.Samples.Kinect.KinectFusionExplorer O   IMicrosoft.Samples.Kinect.KinectFusionExplorer.ReconstructionSensorControl   \ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     @d2http://schemas.microsoft.com/expression/blend/2008     6sys&clr-namespace:System;assembly=mscorlib     WtkHclr-namespace:Microsoft.Kinect.Toolkit;assembly=Microsoft.Kinect.Toolkit     Il;clr-namespace:Microsoft.Samples.Kinect.KinectFusionExplorer       � DataContext  ��  �� RelativeSource �� *Self+ � 	Resources    MediumGreyBrush �� Color  KinectBlueBrush  KinectPurpleBrushH  BMicrosoft.Samples.Kinect.KinectFusionExplorer.DepthToTextConverter  dttP  JMicrosoft.Samples.Kinect.KinectFusionExplorer.FloatValueToIntTextConverter  fvtt �� 
TargetType  KinectRadioButton �� Property �� Value �� 
TargetType �� Orientation	 �Name
 ��Name ��Name �� TextAlignment �� Triggers P� Property P� Value �� 
TargetName  KinectCheckBox  SliderRepeatButtonStyle  HorizontalSliderThumbStyle J� SnapsToDevicePixels ��Name 	 KinectSlider � RowDefinitions m�Name m� Fill m� 	Placement �� CornerRadius ��Name ]�Name ]� DecreaseRepeatButton �� DecreaseLarge ]� IncreaseRepeatButton �� IncreaseLarge ]� Thumb n�Name 
 KinectTextFormat  KinectTextFormatTextBox��%�  &	        &	      &	 *     &	 ?     &	 D     &	 I     7  7  &	 ]    7  7  &	 S    &	 �    7  &		 �	    7  7  7  7  7  7  &	
 U    &	 �    �� $ 	#ff6e6e6e���� $ 	#ff00BCF2���� $ 	#ff52318f��    �� " 	�.���  � �� �� " 	�.R��� $ 
Horizontal=�
������ .+�� $	 RadioButtonChecked��.+��� $
 CheckedNormal��$#��Images\RadioCheckedNormal.png��$
��None=�$��Center=��� $
 CheckedHover��$"��Images\RadioCheckedHover.png��$
��None=�$��Center=�$y�	Collapsed=�� $	 RadioButtonUnchecked��$y�	Collapsed=�.+��� $
 UncheckedNormal��$%��Images\RadioUncheckedNormal.png��$
��None=�$��Center=��� $
 UncheckedHover��$$��Images\RadioUncheckedHover.png��$
��None=�$��Center=�$y�	Collapsed=��� $ RadioButtonText��#��z��$
 Left=�$��Center=�8��  $��15(�$��5 0 0 0q� P�  �@d�	IsChecked$ falseY�. ��� $ RadioButtonChecked�� � y�$ 	Collapsed=��� $ RadioButtonUnchecked�� � y�$ Visible=�P�  �@J�IsMouseOver . . ��� $ CheckedNormal�� � y�$ 	Collapsed=��� $ UncheckedNormal�� � y�$ 	Collapsed=��� $ CheckedHover�� � y�$ Visible=��� $ UncheckedHover�� � y�$ Visible=��� 8  $ RadioButtonText�� � ���� " ��.���  � �� �� " ��.R�� .+��� $ 
Horizontal=�
������ .+�� $	 CheckBoxChecked��.+��� $
 CheckedNormal��$��Images\CheckedNormal.png��$
��None=�$��Center=��� $
 CheckedHover��$��Images\CheckedHover.png��$
��None=�$��Center=�$y�	Collapsed=�� $	 CheckBoxUnchecked��$y�	Collapsed=�.+��� $
 UncheckedNormal��$ ��Images\UncheckedNormal.png��$
��None=�$��Center=��� $
 UncheckedHover��$��Images\UncheckedHover.png��$
��None=�$��Center=�$y�	Collapsed=��� $ CheckBoxText��#��z��$
 Left=�$��Center=�8��  $��15(�$��9,0,0,0q� P�  �@d�	IsChecked$ falseY�. ��� $ CheckBoxChecked�� � y�$ 	Collapsed=��� $ CheckBoxUnchecked�� � y�$ Visible=�P�  �@J�IsMouseOver . . ��� $ CheckedNormal�� � y�$ 	Collapsed=��� $ UncheckedNormal�� � y�$ 	Collapsed=��� $ CheckedHover�� � y�$ Visible=��� $ UncheckedHover�� � y�$ Visible=��� 8  $ CheckBoxText�� � ���� " ��.���  �@�OverridesDefaultStyle . ��  � �� .  ��  � |� .  ��  � �� �� " ��.R��� 
������ �� " n�.���  � |� .  ��  �@�OverridesDefaultStyle . ��  � ��$ 22����  � ��$ 11����  � ��8   ��  � �� �� " n�.R���  . .+��� $
 Body��$��11��$��22��#��z���� " ��.���  �@��IsPressAndHoldEnabled .  ��  � ��
 ���� ��  � ��8   ��  � �� �� " ��.R��� #��z��#��z��#��z�� . ._��  �� $
��Auto ��� $
��Auto �#��z���� $
��Auto �.+�m� $
 Tick��# z��$��4��$	 Top=�$��0��$y�Visible=��� $ TrackBackground��#��z��$ 1��$	��4.0��$	��5,0q�$��1��$��center=�._��� $��-6,-1q�.+��� $ PART_SelectionRange��#��� ��$	��4.0��#��� ��$	��1.0��$y�Hidden=�]� $ 
PART_Track��$��1�� �� #��Z 8��  �� #��Z 8��  n� $ Thumb��8��  P�  �@J�IsMouseOver . . ��� 8   � ���� 8  $ Thumb�� � ���� " ��.���  � ��$ Center=���  � ��$ Center=���  � ��$ Segoe UI)���  � ��$ 15(��� " ��.���  � ��$ Right=���  � ��$ Center=���  � ��$ Segoe UI)���  � ��$ 12(�.���� $��2��$��2��$��2��#��[ $��2q�._��  �� $
��Auto ��� $��44 ��� $
��Auto �  � ColumnDefinitions  �� $��* ��� $��* ��� $��* ��� $��* �.+��� $��4��#��[ $��	-1,-1,0,0q�.+��� #��[
 
�������Image Options�� $ 
Horizontal=�$��1��$��0��$��4��.+��� #��[ $
��Left=�$��Center=�$��10q�$��
Use Camera��4  .System.Windows.Automation.AutomationProperties!   AutomationId$! 	UseSensor��" d� 	IsChecked" �� *	UseSensor+�� -   $��checkBoxNearMode��#��[ $
��Left=�$��Center=�$��10q�$��	Near Mode��$! NearMode��" �� *
NearMode+�� #��[ $
��Left=�$��Center=�$��10q�$��Mirror Depth��$! MirrorDepth��" �� *MirrorDepth+�� #��[ $
��Left=�$��Center=�$��10q�$��Capture Color��$! CaptureColor��" �� *CaptureColor+�� $��2��$��2��$��2��#��[ $��2q�._�� $��2��$��2��$��2�� �� $
��Auto ��� $��* ��� $
��Auto �.+��� $��0��#��[ $��	-1 0 -1 0q�$��20��$	��Top=�.+��� #��[
 
�������Depth Threshold�� $��1��$	��Top=�.+�� $��0 10 0 0q�  �� $
��Auto ��� $��* ��� $
��Auto �.+��� $��0��#��[
 #��[ $��5 0 0 0q�Min�� $��1��#��[	 # � SmallChange$
# 0.02Q�$ � LargeChange$	$ 1.5Q�% � Minimum$
% 0.35Q�& � Maximum$	& 8.0Q�' �� Ticks$2' ,0.35, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0R�$��5q�$! MinDepthClip��( � Value( �� *MinDepthClip+�� $ 
Horizontal=�$��2��$��0 0 0 0q�.+��� #��[ ) �� TextWrapping$) NoWrap=�$��35��$��Right=�#��[ $��0 0 0 0q�.������ * �� Path$* MinDepthClip�+ �� 	Converter#+ [ , �� ConverterParameter, �� - �� Type"- X�.s�X�0.35Q�X�8.0Q�.���� #��[
 #��[ $��2 0 3 0q�$��m���   �� $
��Auto ��� $��* ��� $
��Auto �.+��� $��0��#��[
 #��[ $��5 0 0 0q�Max�� $��1��#��[	 $
# 0.02Q�$	$ 1.5Q�$
% 0.35Q�$	& 8.0Q�$2' ,0.35, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0R�$��5q�$! MaxDepthClip��( �� *MaxDepthClip+�� $ 
Horizontal=�$��2��$��0 0 0 0q�.+��� #��[ $) NoWrap=�$��35��$��Right=�#��[ $��0 0 0 0q�.������ $* MaxDepthClip�#+ [ , �� "- X�.s�X�0.35Q�X�8.0Q�.���� #��[
 #��[ $��2 0 3 0q�$��m���� -   $��2��$��	10 5 10 5q�#��[ 
�������$��2q�$��Reset All To Defaults��$��30��$	��140��
�������|�.  �� $��2��$��0��$��3��$��2��#��[ $��2q�._�� $��0 0 0 0q� �� $��20 ��� $��30 ��� $��30 ��� $��30 ��� $
��Auto �.+��� $��0��$��2��$��4��#��[ $��	-1 0 -1 0q�.+��� #��[
 
�������Camera Pose� $��1��$��0 0 0 -5q�  �� $
��Auto ��� $��* ��� $
��Auto �.+��� $��0��#��[
 #��[ $��5 0 0 0q�Rot X�� $��1��#��[	 $	# 1.0Q�$
$ 45.0Q�$% -180.0Q�$& 180.0Q�$9' 3-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0R�$��5q�$! AngleX��( �� *AngleX+�� $��2��#��[ $) NoWrap=�$��43��$��Right=�#��[ $��0 0 5 0q����� *AngleX+#+ [ � $��2��$��0 0 0 -5q�  �� $
��Auto ��� $��* ��� $
��Auto �.+��� $��0��#��[
 #��[ $��5 0 0 0q�Rot Y�� $��1��#��[	 $	# 1.0Q�$
$ 45.0Q�$% -180.0Q�$& 180.0Q�$9' 3-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0R�$��5q�$! AngleY��( �� *AngleY+�� $��2��#��[ $) NoWrap=�$��43��$��Right=�#��[ $��0 0 5 0q����� *AngleY+#+ [ � $��3��$��0 0 0 -5q�  �� $
��Auto ��� $��* ��� $
��Auto �.+��� $��0��#��[
 #��[ $��5 0 0 0q�Rot Z�� $��1��#��[	 $	# 1.0Q�$
$ 45.0Q�$% -180.0Q�$& 180.0Q�$9' 3-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0R�$��5q�$! AngleZ��( �� *AngleZ+�� $��2��#��[ $) NoWrap=�$��43��$��Right=�#��[ $��0 0 5 0q����� *AngleZ+#+ [ � $��4��$��0 0 0 0q�  �� $
��Auto ��� $��* ��� $
��Auto �.+��� $��0��#��[
 #��[ $��5 0 0 0q�
Distance�� $��1��#��[	 $	# 0.1Q�$	$ 0.5Q�$	% 0.0Q�$	& 6.0Q�$E' ?0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0R�$��5q�$! AxisDistance��( �� *AxisDistance+�� $ 
Horizontal=�$��2��$��0 0 0 0q�.+��� #��[ $) NoWrap=�$��35��$��Right=�#��[ $��0 0 0 0q�.������ $* AxisDistance�#+ [ , �� "- X�.s�X�0.0Q�X�6.0Q�.���� #��[
 #��[ $��2 0 3 0q�$��m��